# --- START OF FILE project/agents/dqn_agents/qr_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import os
import random
import logging

# Import common components (assuming placement in dqn_agents folder)
from networks.tf_networks import build_qr_dqn # Needs to be created
from replay_memory import PER_ReplayBuffer
from agents.base_agent import BaseAgent # Import from parent directory

class QRDQNAgent(BaseAgent):
    """ Quantile Regression DQN (QR-DQN) Agent (TF). """
    def __init__(self, game,
                 # QR-DQN specific parameters
                 num_quantiles=51,
                 kappa=1.0, # Parameter for quantile Huber loss
                 # Standard DQN parameters
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 hidden_units=(64, 64), activation='relu',
                 model_path=None,
                 logger=None):
        super().__init__(game, model_path, logger=logger)

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size

        # QR-DQN parameters
        self.num_quantiles = num_quantiles
        self.kappa = kappa
        # Precompute cumulative quantile midpoints (tau_hat)
        # tau_i = i / N, tau_hat_i = (tau_{i-1} + tau_i) / 2
        # tau_hat[0] = 0.5 / N, tau_hat[i] = (2i+1)/(2N) for i=0..N-1
        self.tau_hat = tf.cast( (tf.range(num_quantiles, dtype=tf.float32) + 0.5) / num_quantiles, dtype=tf.float32) # Shape (N,)

        # Store hyperparameters
        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.per_alpha = per_alpha; self.per_beta_start = per_beta_start; self.per_beta = per_beta_start
        self.per_beta_frames = per_beta_frames

        # Initialize PER buffer
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.per_alpha)
        # frame_idx inherited

        # Build QR-DQN networks
        self.model = build_qr_dqn((self.state_size,), self.action_size, self.num_quantiles,
                                   hidden_units, activation, name="QRDQN_Online")
        self.target_model = build_qr_dqn((self.state_size,), self.action_size, self.num_quantiles,
                                         hidden_units, activation, name="QRDQN_Target")
        self._ensure_model_qr_params(self.model); self._ensure_model_qr_params(self.target_model)

        # Optimizer (Loss calculated manually)
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.update_target_model() # Initial sync

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

        # Epsilon greedy needed for exploration
        self.epsilon_start = 1.0; self.epsilon_end = 0.01; self.epsilon_decay = 0.9999
        self.epsilon = self.epsilon_start

    def _ensure_model_qr_params(self, model_instance):
        """Ensures a model instance has necessary QR attributes."""
        if not hasattr(model_instance, 'num_quantiles'):
            self._log(logging.WARNING, f"Model {model_instance.name} missing QR params. Re-attaching.")
            model_instance.num_quantiles = self.num_quantiles
            model_instance.tau_hat = self.tau_hat

    def remember(self, state, action, reward, next_state, done):
        """Stores experience."""
        self.replay_buffer.add((state.astype(np.float32), action, np.float32(reward),
                                next_state.astype(np.float32), done))

    def get_action(self, state):
        """Select action using epsilon-greedy based on mean of predicted quantiles."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore

        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        # Predict quantiles: shape (1, num_actions, num_quantiles)
        quantile_values = self.model(state_input, training=False).numpy()[0] # Use __call__
        # Calculate expected Q-values (mean of quantiles for each action)
        q_values = np.mean(quantile_values, axis=1) # Mean over quantiles axis -> shape (num_actions,)
        return np.argmax(q_values) # Exploit (action index)

    def update_beta(self):
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.per_beta_frames, 1.0)
        self.per_beta = self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs QR-DQN training step with quantile loss."""
        # --- Calculate Target Quantiles ---
        # Predict next state quantiles with target network: shape (b, na, nq)
        next_quantile_values_target = self.target_model(next_states, training=False)
        # Calculate next state expected Q-values (mean over quantiles) for action selection
        next_q_target = tf.reduce_mean(next_quantile_values_target, axis=2) # (b, na)
        # Select best actions based on target network Q-values (Can use Double-DQN style here if preferred)
        # Simple version: use target net Q for action selection
        next_actions = tf.argmax(next_q_target, axis=1, output_type=tf.int32) # (b,)

        # Gather the quantile values for the selected next actions: shape (b, nq)
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        next_best_quantiles = tf.gather_nd(next_quantile_values_target, gather_indices) # (b, nq)

        # Calculate target quantile values: Tz_j = R + gamma * Z_target_j(s', a*) * (1 - done)
        rewards_exp = tf.expand_dims(tf.cast(rewards, tf.float32), 1) # (b, 1)
        dones_exp = tf.expand_dims(tf.cast(dones, tf.float32), 1)    # (b, 1)
        target_quantiles = rewards_exp + self.gamma * next_best_quantiles * (1.0 - dones_exp) # (b, nq)
        target_quantiles_detached = tf.stop_gradient(target_quantiles) # Shape (b, nq)

        # --- Calculate Loss ---
        with tf.GradientTape() as tape:
            # Predict current state quantiles with online network: shape (b, na, nq)
            current_quantile_values_all = self.model(states, training=True)
            # Gather quantiles for the action actually taken: shape (b, nq)
            action_indices_gather = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_quantiles = tf.gather_nd(current_quantile_values_all, action_indices_gather) # (b, nq)

            # Calculate Quantile Huber Loss
            # Expand dims for broadcasting: current (b, nq, 1), target (b, 1, nq)
            td_error = tf.expand_dims(target_quantiles_detached, axis=1) - tf.expand_dims(current_quantiles, axis=2) # Shape (b, nq, nq)

            # Calculate Huber loss element-wise: L_k(u) = { 0.5 * u^2 if |u| <= k; k * (|u| - 0.5*k) if |u| > k }
            abs_td_error = tf.abs(td_error)
            huber_loss = tf.where(abs_td_error <= self.kappa, 0.5 * tf.square(td_error),
                                  self.kappa * (abs_td_error - 0.5 * self.kappa)) # Shape (b, nq, nq)

            # Calculate Quantile Regression Loss component
            # |tau - (td_error < 0)| * L_k(td_error) / k
            # Indicator (td_error < 0) is 1 if error is negative, 0 otherwise
            indicator = tf.cast(td_error < 0, dtype=tf.float32) # Shape (b, nq, nq)
            # Reshape tau_hat for broadcasting: (1, nq, 1)
            tau_hat_exp = tf.reshape(self.tau_hat, [1, -1, 1]) # Shape (1, nq, 1)
            quantile_weight = tf.abs(tau_hat_exp - indicator) # Shape (b, nq, nq)

            # Element-wise quantile loss: qr_loss = weight * huber_loss
            element_quantile_loss = quantile_weight * huber_loss # Shape (b, nq, nq)

            # Sum over target quantiles (axis 2), then average over predicted quantiles (axis 1)
            # This gives the loss for each sample in the batch
            loss_per_sample = tf.reduce_mean(tf.reduce_sum(element_quantile_loss, axis=2), axis=1) # Shape (b,)

            # Apply PER Importance Sampling weights and calculate final batch mean loss
            loss = tf.reduce_mean(tf.cast(weights, tf.float32) * loss_per_sample) # Scalar loss

        # Compute and Apply Gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Use loss_per_sample as TD error for PER update (before IS weighting and reduction)
        td_errors = loss_per_sample

        return td_errors, loss

    def learn(self):
        """Samples batch, performs QR-DQN training, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.per_beta)
        if batch is None: self._log(logging.WARNING, "QR-DQN sampling failed."); return None

        states_np=np.array([b[0] for b in batch],dtype=np.float32); actions_np=np.array([b[1] for b in batch],dtype=np.int32)
        rewards_np=np.array([b[2] for b in batch],dtype=np.float32); next_states_np=np.array([b[3] for b in batch],dtype=np.float32)
        dones_np=np.array([b[4] for b in batch],dtype=np.float32) # Float for TF

        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np); weights_tf=tf.convert_to_tensor(weights_np,dtype=tf.float32)

        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf)

        td_errors_np = td_errors_tf.numpy() # (b,)
        for i, idx in enumerate(idxs): self.replay_buffer.update(idxs[i], td_errors_np[i])

        if self.epsilon > self.epsilon_end: self.epsilon *= self.epsilon_decay

        metrics = {'loss': loss_tf.numpy(), 'mean_td_error': np.mean(td_errors_np),
                   'beta': self.per_beta, 'epsilon': self.epsilon}
        return metrics

    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"QRDQNAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
            self._ensure_model_qr_params(self.target_model)

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        loaded_model = self._load_keras_model(model_file) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self._ensure_model_qr_params(self.model)
            self.update_target_model() # Syncs target and ensures its params too

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/dqn_agents/qr_dqn_agent.py ---