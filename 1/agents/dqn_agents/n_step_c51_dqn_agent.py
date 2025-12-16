# --- START OF FILE project/agents/dqn_agents/n_step_c51_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import os
import random
from collections import deque
import logging

# Import common components
from networks.tf_networks import build_c51_dqn # C51 network
from replay_memory import PER_ReplayBuffer # Can use PER
from agents.base_agent import BaseAgent

class NStepC51DQNAgent(BaseAgent):
    """ N-Step C51 DQN Agent (TF). """
    def __init__(self, game,
                 # N-Step parameters
                 n_step=3,
                 # C51 parameters
                 num_atoms=51, v_min=-10.0, v_max=10.0,
                 # Standard parameters
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

        # N-Step params
        self.n_step = max(1, n_step)
        self.n_step_buffer = deque(maxlen=self.n_step)

        # C51 parameters
        self.num_atoms = num_atoms; self.v_min = v_min; self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1) if num_atoms > 1 else 0
        self._initialize_support()

        # Store hyperparameters
        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.alpha = per_alpha; self.beta_start = per_beta_start; self.beta = per_beta_start
        self.beta_frames = per_beta_frames

        # Initialize PER buffer (stores N-step transitions)
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.alpha)
        # frame_idx inherited

        # Build C51 networks
        self.model = build_c51_dqn((self.state_size,), self.action_size, self.num_atoms, self.v_min, self.v_max,
                                   hidden_units, activation, name="NStepC51DQN_Online")
        self.target_model = build_c51_dqn((self.state_size,), self.action_size, self.num_atoms, self.v_min, self.v_max,
                                          hidden_units, activation, name="NStepC51DQN_Target")
        self._ensure_model_c51_params(self.model); self._ensure_model_c51_params(self.target_model)

        # Optimizer (Loss calculated manually)
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.update_target_model() # Initial sync

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

        # Epsilon greedy needed
        self.epsilon_start = 1.0; self.epsilon_end = 0.01; self.epsilon_decay = 0.9999
        self.epsilon = self.epsilon_start

    def _initialize_support(self):
        """Initializes the support tensor."""
        self.support = tf.cast(tf.linspace(self.v_min, self.v_max, self.num_atoms), dtype=tf.float32)

    def _ensure_model_c51_params(self, model_instance):
        """Ensures a model instance has necessary C51 attributes."""
        if not hasattr(model_instance, 'support'):
            self._log(logging.WARNING, f"Model {model_instance.name} missing C51 params. Re-attaching.")
            model_instance.num_atoms = self.num_atoms; model_instance.v_min = self.v_min; model_instance.v_max = self.v_max
            model_instance.delta_z = self.delta_z; model_instance.support = self.support

    def _get_n_step_info(self):
        """Calculates N-step reward and identifies state/done N steps away."""
        # Same logic as NStepRainbowAgent
        _state_tn, _action_tn, reward_tn, state_tplusn, done_tplusn = self.n_step_buffer[-1]
        n_step_reward = reward_tn
        n_step_gamma_power = self.gamma
        for i in range(len(self.n_step_buffer) - 2, -1, -1):
            _st, _at, rt, _nst, _dt = self.n_step_buffer[i]
            n_step_reward += n_step_gamma_power * rt
            n_step_gamma_power *= self.gamma
        return n_step_reward, state_tplusn, done_tplusn

    def remember(self, state, action, reward, next_state, done):
        """Stores 1-step transition locally, adds N-step transition to PER buffer."""
        transition = (state.astype(np.float32), action, np.float32(reward),
                      next_state.astype(np.float32), done)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) == self.n_step:
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
            oldest_state, oldest_action, _, _, _ = self.n_step_buffer[0]
            # Store (S_t, A_t, R_n, S_{t+n}, D_{t+n})
            self.replay_buffer.add((oldest_state, oldest_action, n_step_reward, n_step_next_state, n_step_done))

    def get_action(self, state):
        """Select action using epsilon-greedy based on expected Q-values from C51 model."""
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        dist_probs = self.model(state_input, training=False).numpy()[0] # (na, n_atoms)
        if not hasattr(self, 'support') or self.support is None: self._initialize_support()
        support_np = self.support.numpy() # (n_atoms,)
        q_values = np.sum(dist_probs * support_np.reshape(1, -1), axis=1) # (na,)
        return np.argmax(q_values) # Action index

    def update_beta(self):
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    @tf.function
    def _project_n_step_distribution(self, n_step_rewards, n_step_next_states, n_step_dones):
        """Calculates the target C51 distribution for the N-step returns."""
        # Get next state distributions from target network: (b, na, n_atoms)
        next_dist_probs_target = self.target_model(n_step_next_states, training=False)
        # Use online model's expected Q for action selection (Double DQN style)
        next_dist_probs_online = self.model(n_step_next_states, training=False)
        next_q_online = tf.reduce_sum(next_dist_probs_online * self.support, axis=2) # (b, na)
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32) # (b,)

        # Gather the target distribution for the selected next actions: (b, n_atoms)
        batch_indices = tf.range(tf.shape(next_actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        next_best_dist_probs = tf.gather_nd(next_dist_probs_target, gather_indices) # (b, n_atoms)

        # Calculate the N-step discounted support atoms:
        # Tz_j = R_n + (gamma^n) * z_j * (1 - D_{t+n})
        gamma_n = tf.pow(self.gamma, self.n_step)
        rewards_exp = tf.expand_dims(tf.cast(n_step_rewards, tf.float32), 1) # (b, 1)
        dones_exp = tf.expand_dims(tf.cast(n_step_dones, tf.float32), 1)    # (b, 1)
        support_exp = tf.expand_dims(self.support, 0) # (1, n_atoms)

        Tz = rewards_exp + gamma_n * support_exp * (1.0 - dones_exp) # (b, n_atoms)

        # Clamp, calculate bin indices, and distribute probability mass (same as 1-step C51)
        Tz = tf.clip_by_value(Tz, self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = tf.floor(b); u = tf.math.ceil(b)
        l_int = tf.cast(tf.clip_by_value(l, 0.0, float(self.num_atoms - 1)), tf.int32)
        u_int = tf.cast(tf.clip_by_value(u, 0.0, float(self.num_atoms - 1)), tf.int32)
        m_l = next_best_dist_probs * (u - b); m_u = next_best_dist_probs * (b - l)

        # Scatter masses to create target distribution
        target_dist = tf.zeros_like(next_best_dist_probs) # (b, n_atoms)
        batch_idx_mesh = tf.tile(tf.expand_dims(batch_indices, 1), [1, self.num_atoms])
        indices_l = tf.stack([tf.reshape(batch_idx_mesh, [-1]), tf.reshape(l_int, [-1])], axis=1)
        target_dist = tf.tensor_scatter_nd_add(target_dist, indices_l, tf.reshape(m_l, [-1]))
        indices_u = tf.stack([tf.reshape(batch_idx_mesh, [-1]), tf.reshape(u_int, [-1])], axis=1)
        target_dist = tf.tensor_scatter_nd_add(target_dist, indices_u, tf.reshape(m_u, [-1]))

        return tf.stop_gradient(target_dist) # Return detached target distribution

    @tf.function
    def _train_step_tf(self, states, actions, n_step_rewards, n_step_next_states, n_step_dones, weights):
        """Performs N-Step C51 training step."""
        # Calculate the target distribution based on N-step info
        target_distribution = self._project_n_step_distribution(n_step_rewards, n_step_next_states, n_step_dones)

        # Calculate Loss using online network predictions
        with tf.GradientTape() as tape:
            # Get current distribution predictions from online model for actions taken
            current_dist_all = self.model(states, training=True) # (b, na, n_atoms)
            batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            action_indices_gather = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_dist = tf.gather_nd(current_dist_all, action_indices_gather) # (b, n_atoms)

            # Calculate Cross-entropy loss against the N-step target distribution
            cross_entropy = -tf.reduce_sum(target_distribution * tf.math.log(current_dist + 1e-7), axis=1) # Sum over atoms -> (b,)
            # Apply PER IS weights and reduce
            loss = tf.reduce_mean(tf.cast(weights, tf.float32) * cross_entropy) # Mean over batch

        # Compute and Apply Gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Use cross-entropy loss as TD error for PER update
        td_errors = cross_entropy
        return td_errors, loss

    def learn(self):
        """Samples N-step batch, performs training, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()
        # Sample N-step transitions: (S_t, A_t, R_n, S_{t+n}, D_{t+n})
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None: self._log(logging.WARNING, "N-Step C51 sampling failed."); return None

        states_np = np.array([b[0] for b in batch], dtype=np.float32) # S_t
        actions_np = np.array([b[1] for b in batch], dtype=np.int32)  # A_t
        n_rewards_np = np.array([b[2] for b in batch], dtype=np.float32) # R_n
        n_next_states_np = np.array([b[3] for b in batch], dtype=np.float32) # S_{t+n}
        n_dones_np = np.array([b[4] for b in batch], dtype=np.float32) # D_{t+n}

        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        n_rewards_tf=tf.convert_to_tensor(n_rewards_np); n_next_states_tf=tf.convert_to_tensor(n_next_states_np)
        n_dones_tf=tf.convert_to_tensor(n_dones_np); weights_tf=tf.convert_to_tensor(weights_np,dtype=tf.float32)

        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, n_rewards_tf, n_next_states_tf, n_dones_tf, weights_tf)

        td_errors_np = td_errors_tf.numpy() # (b,)
        for i, idx in enumerate(idxs): self.replay_buffer.update(idxs[i], td_errors_np[i])

        if self.epsilon > self.epsilon_end: self.epsilon *= self.epsilon_decay

        metrics = {'loss': loss_tf.numpy(), 'mean_td_error': np.mean(td_errors_np),
                   'beta': self.beta, 'epsilon': self.epsilon}
        return metrics

    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"NStepC51DQNAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
            self._ensure_model_c51_params(self.target_model)

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        loaded_model = self._load_keras_model(model_file) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self._ensure_model_c51_params(self.model)
            self.update_target_model() # Syncs target and ensures its params too

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/dqn_agents/n_step_c51_dqn_agent.py ---