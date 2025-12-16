# --- START OF FILE project/agents/dqn_agents/dueling_soft_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import logging

# Import common components
from networks.tf_networks import build_dueling_dqn
from replay_memory import PER_ReplayBuffer
from layers.custom_layers_tf import NoisyLinear, MeanReducer # Needed for loading
from agents.base_agent import BaseAgent

class DuelingSoftDQNAgent(BaseAgent):
    """ Dueling Double DQN Agent with Soft Target Updates (TF). """
    def __init__(self, game,
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000,
                 target_update_tau=0.005, # Soft update factor << 1.0
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 noisy_fc=True, noisy_std_init=0.5, # Using Noisy Nets like Rainbow
                 shared_hidden_units=(128, 128), activation='relu',
                 model_path=None,
                 logger=None):
        super().__init__(game, model_path, logger=logger)

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size

        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size
        self.tau = target_update_tau # Soft update factor

        # PER parameters
        self.alpha = per_alpha; self.beta_start = per_beta_start; self.beta = per_beta_start
        self.beta_frames = per_beta_frames; self.noisy_fc = noisy_fc

        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.alpha)
        # frame_idx inherited

        # Build Networks (using Noisy for Rainbow features)
        self.model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=self.noisy_fc, noisy_std_init=noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="DuelingSoftDQN_Online")
        self.target_model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=self.noisy_fc, noisy_std_init=noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="DuelingSoftDQN_Target")

        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE) # Element-wise loss

        # Initial hard sync of target model
        self.target_model.set_weights(self.model.get_weights())
        self._log(logging.INFO, "Initial hard sync of target network complete.")

        if self.model_path_base:
            self.load(self.model_path_base) # Loads online, target is synced after load

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the PER buffer."""
        self.replay_buffer.add((state.astype(np.float32), action, np.float32(reward),
                                next_state.astype(np.float32), done))

    def get_action(self, state):
        """Returns action based on noisy network prediction."""
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        q_values = self.model(state_input, training=self.noisy_fc).numpy() # Use noise if enabled
        return np.argmax(q_values[0])

    def update_beta(self):
        """Anneals beta linearly."""
        fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    # Training step is standard Double Dueling DQN
    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs the core training calculations (DDQN, PER loss). Returns TD errors and loss."""
        # Double DQN Target Calculation
        online_next_q = self.model(next_states, training=self.noisy_fc) # Use noise if enabled
        next_actions = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        # Target uses its own weights (potentially noisy if enabled, but often disabled for target)
        target_next_q = self.target_model(next_states, training=False) # Typically no noise for target

        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)
        target = tf.cast(rewards, tf.float32) + self.gamma * ddqn_next_val * (1.0 - tf.cast(dones, tf.float32))

        # Loss Calculation and Gradients
        with tf.GradientTape() as tape:
            current_q_all = self.model(states, training=self.noisy_fc) # Online model needs noise if enabled
            action_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            element_loss = self.loss_fn(target, current_q)
            weighted_element_loss = tf.cast(weights, tf.float32) * element_loss
            loss = tf.reduce_mean(weighted_element_loss)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        td_errors = tf.abs(target - current_q)
        return td_errors, loss

    def learn(self):
        """Samples batch, performs training step, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None: self._log(logging.WARNING, "Dueling Soft DQN sampling failed."); return None

        states_np=np.array([b[0] for b in batch],dtype=np.float32); actions_np=np.array([b[1] for b in batch],dtype=np.int32)
        rewards_np=np.array([b[2] for b in batch],dtype=np.float32); next_states_np=np.array([b[3] for b in batch],dtype=np.float32)
        dones_np=np.array([b[4] for b in batch],dtype=np.float32) # Float for TF

        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np); weights_tf=tf.convert_to_tensor(weights_np,dtype=tf.float32)

        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf)

        td_errors_np = td_errors_tf.numpy()
        for i, idx in enumerate(idxs): self.replay_buffer.update(idx, td_errors_np[i])

        # --- Perform Soft Target Update ---
        self.update_target_model() # Call soft update after each learn step

        metrics = {'loss': loss_tf.numpy(), 'mean_td_error': np.mean(td_errors_np), 'beta': self.beta}
        return metrics

    def update_target_model(self):
        """Performs soft update (Polyak averaging) of target network weights."""
        if self.model is None or self.target_model is None:
            self._log(logging.WARNING, "Attempted soft update with uninitialized models.")
            return

        online_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_target_weights = []

        if len(online_weights) != len(target_weights):
             self._log(logging.ERROR, "Online and target model weight counts differ. Skipping soft update.")
             return

        for online_w, target_w in zip(online_weights, target_weights):
            # Ensure tensors/arrays are compatible for the operation
            try:
                new_w = self.tau * online_w + (1.0 - self.tau) * target_w
                new_target_weights.append(new_w)
            except Exception as e:
                 self._log(logging.ERROR, f"Error during soft update calculation for a weight: {e}. Skipping update.", exc_info=True)
                 return # Stop update if calculation fails for any weight

        try:
            self.target_model.set_weights(new_target_weights)
            self._log(logging.DEBUG, f"Soft target update performed (tau={self.tau}).")
        except Exception as e:
            self._log(logging.ERROR, f"Error setting soft-updated weights: {e}", exc_info=True)

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        custom_objects = {}
        if self.noisy_fc: custom_objects.update({'NoisyLinear': NoisyLinear, 'MeanReducer': MeanReducer})
        loaded_model = self._load_keras_model(model_file, custom_objects) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            # After loading online model, immediately sync target model hard
            self.target_model.set_weights(self.model.get_weights())
            self._log(logging.INFO, "Hard sync of target network after loading online model.")

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/dqn_agents/dueling_soft_dqn_agent.py ---