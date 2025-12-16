# --- START OF FILE project/agents/dqn_rainbow_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import logging # Import logging

# Import network builder, PER buffer, custom layers, and base agent
from networks.tf_networks import build_dueling_dqn
from replay_memory import PER_ReplayBuffer
from layers.custom_layers_tf import NoisyLinear, MeanReducer # Needed for loading
from agents.base_agent import BaseAgent

class DQNRainbowAgent(BaseAgent):
    """ Rainbow DQN Agent (TF) - Uses network builder """
    def __init__(self, game,
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 noisy_std_init=0.5,
                 shared_hidden_units=(128, 128), activation='relu', # Network params
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size
        self.action_size = game.action_size

        self.gamma = gamma
        self.learning_rate = lr # Store learning rate if needed later
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq # In frames

        # PER parameters
        self.alpha = per_alpha
        self.beta_start = per_beta_start
        self.beta = per_beta_start
        self.beta_frames = per_beta_frames

        # Noisy Nets parameter
        self.noisy_std_init = noisy_std_init

        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.alpha)
        # frame_idx inherited

        # Build Networks using Noisy Nets
        self.model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=True, noisy_std_init=self.noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="DQNRainbow_Online")
        self.target_model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=True, noisy_std_init=self.noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="DQNRainbow_Target")

        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.update_target_model()

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the PER buffer."""
        self.replay_buffer.add((state.astype(np.float32),
                                action,
                                np.float32(reward),
                                next_state.astype(np.float32),
                                done))

    def get_action(self, state):
        """Returns action based on noisy network prediction."""
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        # Use model call with training=True to sample noise
        q_values = self.model(state_input, training=True).numpy()
        return np.argmax(q_values[0])

    def update_beta(self):
        """Anneals beta linearly."""
        fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs the core training calculations (DDQN, PER loss). Returns TD errors and loss."""
        # Double DQN Target Calculation
        online_next_q = self.model(next_states, training=True) # Use noise for action selection
        next_actions = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        target_next_q = self.target_model(next_states, training=False) # Use target without noise

        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)
        target = tf.cast(rewards, tf.float32) + self.gamma * ddqn_next_val * (1.0 - tf.cast(dones, tf.float32))

        # Loss Calculation and Gradients
        with tf.GradientTape() as tape:
            current_q_all = self.model(states, training=True) # Online model needs noise
            action_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            # Calculate element-wise loss
            element_loss = self.loss_fn(target, current_q)
            # Apply IS weights before reduction
            weighted_element_loss = tf.cast(weights, tf.float32) * element_loss
            # Reduce to scalar loss for optimizer
            loss = tf.reduce_mean(weighted_element_loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Calculate TD errors for PER update
        td_errors = tf.abs(target - current_q)
        # Return TD errors and the calculated scalar loss
        return td_errors, loss

    def learn(self):
        """Samples batch, performs training step, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None:
            self._log(logging.WARNING, "Rainbow sampling failed.")
            return None

        states_np = np.array([b[0] for b in batch], dtype=np.float32)
        actions_np = np.array([b[1] for b in batch], dtype=np.int32)
        rewards_np = np.array([b[2] for b in batch], dtype=np.float32)
        next_states_np = np.array([b[3] for b in batch], dtype=np.float32)
        dones_np = np.array([b[4] for b in batch], dtype=np.float32) # Float for TF function

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
        self._log(logging.INFO, f"DQNRainbowAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        # Required custom objects for Dueling + NoisyLinear
        custom_objects = {'NoisyLinear': NoisyLinear, 'MeanReducer': MeanReducer}
        loaded_model = self._load_keras_model(model_file, custom_objects) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self.update_target_model()

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        # Required custom objects must be available when saving if model uses them,
        # but Keras save usually handles registered custom layers automatically.
        # self._save_keras_model includes the optimizer state.
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/dqn_rainbow_agent.py ---
