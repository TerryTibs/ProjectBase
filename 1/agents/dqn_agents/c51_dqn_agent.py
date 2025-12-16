# --- START OF FILE project/agents/c51_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import os
import random
import logging # Import logging

# Import common components
from networks.tf_networks import build_c51_dqn # Specific network builder
from replay_memory import PER_ReplayBuffer # Can use PER with C51
from agents.base_agent import BaseAgent

class C51DQNAgent(BaseAgent):
    """ Categorical DQN (C51) Agent (TF). Learns distributions over Q-values. """
    def __init__(self, game,
                 num_atoms=51, v_min=-10.0, v_max=10.0, # C51 specific params
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 hidden_units=(64, 64), activation='relu', # Network params
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size

        # C51 parameters
        self.num_atoms = num_atoms; self.v_min = v_min; self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1) if num_atoms > 1 else 0
        # Ensure support is created correctly even if loaded model lacks it later
        self._initialize_support()

        # Store hyperparameters
        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.alpha = per_alpha; self.beta_start = per_beta_start; self.beta = per_beta_start
        self.beta_frames = per_beta_frames

        # Initialize PER buffer
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.alpha)
        # frame_idx inherited

        # Build C51 networks
        self.model = build_c51_dqn(
            (self.state_size,), self.action_size, self.num_atoms, self.v_min, self.v_max,
            hidden_units, activation, name="C51DQN_Online")
        self.target_model = build_c51_dqn(
            (self.state_size,), self.action_size, self.num_atoms, self.v_min, self.v_max,
            hidden_units, activation, name="C51DQN_Target")
        # Ensure target model also has C51 attributes
        self._ensure_model_c51_params(self.model)
        self._ensure_model_c51_params(self.target_model)

        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        # Loss calculated manually in _train_step_tf
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
        """Ensures a model instance has the necessary C51 attributes."""
        if not hasattr(model_instance, 'support'):
            self._log(logging.WARNING, f"Model {model_instance.name} missing C51 params. Re-attaching.")
            model_instance.num_atoms = self.num_atoms
            model_instance.v_min = self.v_min
            model_instance.v_max = self.v_max
            model_instance.delta_z = self.delta_z
            model_instance.support = self.support


    def remember(self, state, action, reward, next_state, done):
        """Stores experience."""
        self.replay_buffer.add((state.astype(np.float32),
                                action,
                                np.float32(reward),
                                next_state.astype(np.float32),
                                done))

    def get_action(self, state):
        """Select action using epsilon-greedy based on expected Q-values."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore

        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        # Predict distribution probabilities: shape (1, num_actions, num_atoms)
        dist_probs = self.model(state_input, training=False).numpy()[0] # Use __call__
        # Calculate expected Q-values: Q(s,a) = Σ_i z_i * p_i(s,a)
        # Ensure self.support is available and has the right shape
        if not hasattr(self, 'support') or self.support is None:
             self._initialize_support()
        support_np = self.support.numpy() # (num_atoms,)
        # Reshape support for broadcasting: (1, num_atoms)
        q_values = np.sum(dist_probs * support_np.reshape(1, -1), axis=1) # Sum over atoms axis (axis=1)
        return np.argmax(q_values) # Exploit (action index)


    def update_beta(self):
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs C51 training step with target distribution projection."""
        # Ensure models have C51 params - needed within tf.function context? Maybe not if attached outside.
        # Let's assume they are attached correctly during __init__ and load.

        # Calculate Target Distribution using target network
        next_dist_probs_target = self.target_model(next_states, training=False) # (b, na, n_atoms)
        # Calculate expected Q-values from target distribution to select next actions (DDQN style)
        # Use online model's expected Q for action selection to reduce maximization bias
        next_dist_probs_online = self.model(next_states, training=False) # (b, na, n_atoms)
        next_q_online = tf.reduce_sum(next_dist_probs_online * self.support, axis=2) # (b, na) - Use self.support
        next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32) # (b,) - Best actions from online net Q

        # Gather the target distribution for the selected next actions
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        next_best_dist_probs = tf.gather_nd(next_dist_probs_target, gather_indices) # (b, n_atoms)

        # Project Target Distribution
        rewards = tf.cast(rewards, tf.float32); dones = tf.cast(dones, tf.float32)
        # Calculate projected support atoms Tz = R + gamma * z * (1-done)
        Tz = tf.expand_dims(rewards, 1) + self.gamma * tf.expand_dims(self.support, 0) * (1.0 - tf.expand_dims(dones, 1))
        # If done, target is just the reward R projected onto the atoms (value is 0 after termination)
        # This projection is tricky - simplest is to clamp reward to Vmin/Vmax and create a dirac delta?
        # Paper method: Clamp the projected support Tz
        Tz = tf.clip_by_value(Tz, self.v_min, self.v_max) # (b, n_atoms)

        # Calculate bin indices (b), lower (l), upper (u)
        b = (Tz - self.v_min) / self.delta_z # (b, n_atoms) - fractional bin index
        l = tf.floor(b)
        u = tf.math.ceil(b)

        # Ensure indices are integers and within valid range [0, num_atoms - 1]
        l_int = tf.cast(tf.clip_by_value(l, 0.0, float(self.num_atoms - 1)), tf.int32)
        u_int = tf.cast(tf.clip_by_value(u, 0.0, float(self.num_atoms - 1)), tf.int32)

        # Distribute probability mass (dL = u-b, dU = b-l)
        m_l = next_best_dist_probs * (u - b) # Mass to lower bin
        m_u = next_best_dist_probs * (b - l) # Mass to upper bin

        # Create target distribution matrix P(s', a*) by scattering masses
        target_dist = tf.zeros_like(next_best_dist_probs) # (b, n_atoms)
        # Create indices for tf.tensor_scatter_nd_add
        # Shape: (b * n_atoms, 2) where each row is [batch_index, atom_index]
        batch_idx_mesh = tf.tile(tf.expand_dims(batch_indices, 1), [1, self.num_atoms])
        # Scatter lower bin contributions
        indices_l = tf.stack([tf.reshape(batch_idx_mesh, [-1]), tf.reshape(l_int, [-1])], axis=1)
        target_dist = tf.tensor_scatter_nd_add(target_dist, indices_l, tf.reshape(m_l, [-1]))
        # Scatter upper bin contributions
        indices_u = tf.stack([tf.reshape(batch_idx_mesh, [-1]), tf.reshape(u_int, [-1])], axis=1)
        target_dist = tf.tensor_scatter_nd_add(target_dist, indices_u, tf.reshape(m_u, [-1]))
        # Detach target distribution from gradient computation
        target_dist = tf.stop_gradient(target_dist)

        # Calculate Loss using online network predictions
        with tf.GradientTape() as tape:
            # Get current distribution predictions from the online model for all actions
            current_dist_all = self.model(states, training=True) # (b, na, n_atoms)
            # Gather the distribution for the specific action taken
            action_indices_gather = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_dist = tf.gather_nd(current_dist_all, action_indices_gather) # (b, n_atoms)

            # Calculate Cross-entropy loss: D_KL( P || Q ) often simplified to -sum(P * log(Q))
            # Add epsilon for numerical stability inside log
            cross_entropy = -tf.reduce_sum(target_dist * tf.math.log(current_dist + 1e-7), axis=1) # Sum over atoms -> (b,)

            # Apply PER Importance Sampling weights
            # Ensure weights are float32 and have compatible shape (b,)
            loss = tf.reduce_mean(tf.cast(weights, tf.float32) * cross_entropy) # Mean over batch

        # Compute and Apply Gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Use cross-entropy loss as TD error for PER update (before IS weighting and reduction)
        td_errors = cross_entropy
        # Return TD errors and the scalar loss
        return td_errors, loss


    def learn(self):
        """Samples batch, performs C51 training step, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None:
            self._log(logging.WARNING, "C51 sampling failed.")
            return None

        # Prepare batch data
        states_np = np.array([b[0] for b in batch], dtype=np.float32)
        actions_np = np.array([b[1] for b in batch], dtype=np.int32)
        rewards_np = np.array([b[2] for b in batch], dtype=np.float32)
        next_states_np = np.array([b[3] for b in batch], dtype=np.float32)
        dones_np = np.array([b[4] for b in batch], dtype=np.float32) # Float for TF

        # Convert to tensors
        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np); weights_tf=tf.convert_to_tensor(weights_np,dtype=tf.float32)

        # Execute train step
        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf)

        # Update priorities
        td_errors_np = td_errors_tf.numpy() # (b,)
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idxs[i], td_errors_np[i])

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # Return metrics
        metrics = {
            'loss': loss_tf.numpy(),
            'mean_td_error': np.mean(td_errors_np), # TD error here is cross-entropy
            'beta': self.beta,
            'epsilon': self.epsilon
        }
        return metrics

    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"C51DQNAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
            # Ensure target model also has params after weight copy? Weights are copied, but custom attrs are not.
            self._ensure_model_c51_params(self.target_model)

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        # C51 uses standard Dense layers, no custom objects needed by default
        loaded_model = self._load_keras_model(model_file) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            # Re-attach C51 params if they weren't saved with model config
            self._ensure_model_c51_params(self.model)
            self.update_target_model() # Syncs target and ensures its params too

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        # Note: C51 parameters (support, v_min, etc.) are NOT part of Keras standard save.
        # They need to be re-attached during __init__ or load.
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/c51_dqn_agent.py ---
