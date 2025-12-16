# --- START OF FILE project/agents/n_step_rainbow_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
from collections import deque
import logging # Import logging

# Import common components
from networks.tf_networks import build_dueling_dqn
from replay_memory import PER_ReplayBuffer
from layers.custom_layers_tf import NoisyLinear, MeanReducer # For loading
from agents.base_agent import BaseAgent

class NStepRainbowAgent(BaseAgent):
    """ N-Step Rainbow DQN Agent (TF). """
    def __init__(self, game,
                 n_step=3, # N-step return parameter
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 noisy_std_init=0.5,
                 shared_hidden_units=(128, 128), activation='relu',
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size

        self.n_step = max(1, n_step) # Ensure n_step >= 1
        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq

        self.alpha = per_alpha; self.beta_start = per_beta_start; self.beta = per_beta_start
        self.beta_frames = per_beta_frames; self.noisy_std_init = noisy_std_init

        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.alpha)
        # Buffer to hold the last N transitions for N-step calculation
        self.n_step_buffer = deque(maxlen=self.n_step)
        # frame_idx inherited

        # Build networks (same as Rainbow DQN)
        self.model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=True, noisy_std_init=self.noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="NStepRainbow_Online")
        self.target_model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=True, noisy_std_init=self.noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="NStepRainbow_Target")

        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE) # Element-wise Huber
        self.update_target_model() # Initial sync

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

    def _get_n_step_info(self):
        """Calculates the N-step discounted reward and the state/done N steps away."""
        # Calculate N-step discounted return G_t^n = R_{t+1} + gamma*R_{t+2} + ... + gamma^(n-1)*R_{t+n}
        # And find the state S_{t+n} and done D_{t+n}

        # Get the last reward (R_{t+n}) and done state (D_{t+n}) from the buffer
        _state_tn, _action_tn, reward_tn, state_tplusn, done_tplusn = self.n_step_buffer[-1]
        n_step_reward = reward_tn
        n_step_gamma = self.gamma # Keep track of gamma^k

        # Iterate backwards from the second-to-last transition (index -2 down to 0)
        for i in range(len(self.n_step_buffer) - 2, -1, -1):
            _state_tk, _action_tk, reward_tk, _next_state_tk, _done_tk = self.n_step_buffer[i]
            n_step_reward += n_step_gamma * reward_tk
            n_step_gamma *= self.gamma # Increase power of gamma

        # The state N steps away (S_{t+n}) and its done status (D_{t+n}) are from the last transition in the buffer
        return n_step_reward, state_tplusn, done_tplusn

    def remember(self, state, action, reward, next_state, done):
        """Stores the 1-step transition and adds the oldest N-step transition to PER buffer."""
        # Ensure data types are consistent for storage
        transition = (state.astype(np.float32),
                      action,
                      np.float32(reward),
                      next_state.astype(np.float32),
                      done)
        self.n_step_buffer.append(transition)

        # If buffer has N steps, process the oldest N-step transition
        if len(self.n_step_buffer) == self.n_step:
            # Calculate the N-step return and the state/done N steps ahead
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()

            # Get the state and action from the *oldest* transition in the buffer (S_t, A_t)
            oldest_state, oldest_action, _, _, _ = self.n_step_buffer[0]

            # Add the N-step experience (S_t, A_t, R_n, S_{t+n}, D_{t+n}) to the main PER buffer
            self.replay_buffer.add((oldest_state, oldest_action, n_step_reward, n_step_next_state, n_step_done))

    def get_action(self, state):
        """Selects action using the noisy online network."""
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        # Use training=True to sample noise from NoisyLinear layers
        q_values = self.model(state_input, training=True).numpy()
        return np.argmax(q_values[0])

    def update_beta(self):
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, n_step_rewards, n_step_next_states, n_step_dones, weights):
        """Performs training step using N-step returns."""
        # N-Step Double DQN Target Calculation
        # Action selection using online model (with noise)
        online_next_q = self.model(n_step_next_states, training=True)
        next_actions = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        # Value estimation using target model (without noise)
        target_next_q = self.target_model(n_step_next_states, training=False)

        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        n_step_ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)

        # Calculate N-step discounted Q target
        # Target = R_n + (gamma^n) * Q_target(S_{t+n}, argmax_a Q_online(S_{t+n}, a)) * (1 - D_{t+n})
        gamma_n = tf.pow(self.gamma, self.n_step) # Discount factor to the power of N
        target = tf.cast(n_step_rewards, tf.float32) + \
                 gamma_n * n_step_ddqn_next_val * (1.0 - tf.cast(n_step_dones, tf.float32))

        # Loss Calculation and Gradient Update
        with tf.GradientTape() as tape:
            # Get Q-values for current states (S_t) using online model (with noise)
            current_q_all = self.model(states, training=True)
            # Gather Q-values for the actions actually taken (A_t)
            action_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            # Calculate element-wise Huber loss
            element_loss = self.loss_fn(target, current_q)
            # Apply PER Importance Sampling weights before reduction
            weighted_element_loss = tf.cast(weights, tf.float32) * element_loss
            # Reduce to scalar loss
            loss = tf.reduce_mean(weighted_element_loss)

        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Calculate TD errors for PER update (based on N-step target)
        td_errors = tf.abs(target - current_q)
        # Return TD errors and loss
        return td_errors, loss

    def learn(self):
        """Samples N-step transitions from PER buffer and trains."""
        # Check if enough N-step transitions are in the main buffer
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()

        # Sample N-step transitions (s_t, a_t, R_n, s_{t+n}, d_{t+n}) from PER buffer
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None:
            self._log(logging.WARNING,"NStep Rainbow sampling failed.")
            return None

        # Unpack N-step data
        states_np = np.array([b[0] for b in batch], dtype=np.float32) # S_t
        actions_np = np.array([b[1] for b in batch], dtype=np.int32)  # A_t
        n_rewards_np = np.array([b[2] for b in batch], dtype=np.float32) # R_n
        n_next_states_np = np.array([b[3] for b in batch], dtype=np.float32) # S_{t+n}
        n_dones_np = np.array([b[4] for b in batch], dtype=np.float32) # D_{t+n}

        # Convert to tensors
        states_tf = tf.convert_to_tensor(states_np); actions_tf = tf.convert_to_tensor(actions_np)
        n_rewards_tf = tf.convert_to_tensor(n_rewards_np); n_next_states_tf = tf.convert_to_tensor(n_next_states_np)
        n_dones_tf = tf.convert_to_tensor(n_dones_np); weights_tf = tf.convert_to_tensor(weights_np, dtype=tf.float32)

        # Use the specific N-step training function
        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, n_rewards_tf, n_next_states_tf, n_dones_tf, weights_tf)

        # Update PER priorities
        td_errors_np = td_errors_tf.numpy()
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idx, td_errors_np[i])

        # Return metrics
        metrics = {
            'loss': loss_tf.numpy(),
            'mean_td_error': np.mean(td_errors_np),
            'beta': self.beta
        }
        return metrics


    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"NStepRainbowAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        # Custom objects needed for Dueling + Noisy
        custom_objects = {'NoisyLinear': NoisyLinear, 'MeanReducer': MeanReducer}
        loaded_model = self._load_keras_model(model_file, custom_objects) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self.update_target_model()

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/n_step_rainbow_agent.py ---
