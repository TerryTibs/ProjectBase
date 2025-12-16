# --- START OF FILE project/agents/dqn_lstm_boltze_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import random
import os
import logging # Import logging

# Import network builder, PER buffer, custom layers (MeanReducer only), and base agent
from networks.tf_networks import build_lstm_dueling_dqn
from replay_memory import PER_ReplayBuffer
from layers.custom_layers_tf import MeanReducer # Needed for loading models
from agents.base_agent import BaseAgent

class DQNLSTMBoltzeAgent(BaseAgent):
    """ LSTM DQN Agent with Boltzmann Exploration (TF) - Uses network builder """
    def __init__(self, game,
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 lstm_units=32,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 temperature=1.0, # Boltzmann temperature
                 shared_hidden_units=(64, 64), activation='relu', # Network params
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size
        self.action_size = game.action_size

        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.lstm_units = lstm_units; self.temperature = temperature # Store temperature

        # PER parameters
        self.alpha = per_alpha; self.beta_start = per_beta_start; self.beta = per_beta_start
        self.beta_frames = per_beta_frames
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.alpha)
        # frame_idx inherited

        # Build Networks using Dense layers (noisy=False)
        self.lstm_input_shape = (1, self.state_size)
        self.model = build_lstm_dueling_dqn(
            input_shape=self.lstm_input_shape, num_actions=self.action_size, lstm_units=self.lstm_units,
            noisy=False, # Specify Dense layers for this agent
            shared_hidden_units=shared_hidden_units, activation=activation, name="DQNLSTMBoltze_Online")
        self.target_model = build_lstm_dueling_dqn(
             input_shape=self.lstm_input_shape, num_actions=self.action_size, lstm_units=self.lstm_units,
             noisy=False, shared_hidden_units=shared_hidden_units, activation=activation, name="DQNLSTMBoltze_Target")

        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.update_target_model()

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

    def remember(self, state, action, reward, next_state, done):
        """Stores flat state vectors in the PER buffer."""
        self.replay_buffer.add((state.astype(np.float32),
                                action,
                                np.float32(reward),
                                next_state.astype(np.float32),
                                done))

    def get_action(self, state):
        """Returns action using Boltzmann exploration."""
        state_input = np.reshape(np.asarray(state, dtype=np.float32), [1, 1, self.state_size])
        # No training=True needed for Dense layers in predict
        q_values = self.model.predict(state_input, verbose=0)[0]

        temp = max(self.temperature, 1e-8) # Avoid zero temperature
        # Calculate probabilities using softmax with temperature
        exp_q = np.exp(q_values / temp)
        sum_exp_q = np.sum(exp_q)

        # Handle potential numerical instability (sum=0 or inf)
        if sum_exp_q <= 1e-8 or not np.isfinite(sum_exp_q):
             # Fallback to uniform distribution if calculation fails
             self._log(logging.DEBUG, f"Boltzmann calculation unstable (Sum={sum_exp_q}). Using uniform probabilities.")
             probabilities = np.ones(self.action_size) / self.action_size
        else:
            probabilities = exp_q / sum_exp_q

        # Sample action based on calculated probabilities
        action = np.random.choice(self.action_size, p=probabilities)
        return action

    def update_beta(self):
        """Anneals beta linearly."""
        fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs the core training calculations (DDQN, PER loss) with LSTM."""
        # Reshape flat states for LSTM
        states_lstm = tf.reshape(states, [-1, 1, self.state_size])
        next_states_lstm = tf.reshape(next_states, [-1, 1, self.state_size])

        # Double DQN Target Calculation (using Dense layers)
        online_next_q = self.model(next_states_lstm, training=False) # No training=True needed for Dense
        next_actions = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        target_next_q = self.target_model(next_states_lstm, training=False)

        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)
        target = tf.cast(rewards, tf.float32) + self.gamma * ddqn_next_val * (1.0 - tf.cast(dones, tf.float32))

        # Loss Calculation and Gradients
        with tf.GradientTape() as tape:
            current_q_all = self.model(states_lstm, training=True) # Need training=True for gradient calc
            action_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            # Element-wise Huber loss
            element_loss = self.loss_fn(target, current_q)
            # Apply IS weights before reduction
            weighted_element_loss = tf.cast(weights, tf.float32) * element_loss
            # Reduce to scalar loss
            loss = tf.reduce_mean(weighted_element_loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Calculate TD errors for PER update
        td_errors = tf.abs(target - current_q)
        # Return TD errors and loss
        return td_errors, loss


    def learn(self):
        """Samples batch, performs training step, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()

        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None:
            self._log(logging.WARNING, "Boltze LSTM sampling failed.")
            return None

        states_np = np.array([b[0] for b in batch], dtype=np.float32)
        actions_np = np.array([b[1] for b in batch], dtype=np.int32)
        rewards_np = np.array([b[2] for b in batch], dtype=np.float32)
        next_states_np = np.array([b[3] for b in batch], dtype=np.float32)
        dones_np = np.array([b[4] for b in batch], dtype=np.float32) # Float for TF

        # Convert to tensors
        states_tf = tf.convert_to_tensor(states_np); actions_tf = tf.convert_to_tensor(actions_np)
        rewards_tf = tf.convert_to_tensor(rewards_np); next_states_tf = tf.convert_to_tensor(next_states_np)
        dones_tf = tf.convert_to_tensor(dones_np); weights_tf = tf.convert_to_tensor(weights_np, dtype=tf.float32)

        # Execute train step and get metrics
        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf)

        # Update PER priorities
        td_errors_np = td_errors_tf.numpy()
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idx, td_errors_np[i])

        # Prepare metrics dictionary
        metrics = {
            'loss': loss_tf.numpy(),
            'mean_td_error': np.mean(td_errors_np),
            'beta': self.beta
        }
        return metrics


    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"DQNLSTMBoltzeAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        # Only MeanReducer needed if Dueling architecture was used
        custom_objects = {'MeanReducer': MeanReducer}
        loaded_model = self._load_keras_model(model_file, custom_objects) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self.update_target_model()

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/dqn_lstm_boltze_agent.py ---
