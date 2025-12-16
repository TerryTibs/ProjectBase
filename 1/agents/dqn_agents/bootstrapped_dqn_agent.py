# --- START OF FILE project/agents/bootstrapped_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import random
import logging # Import logging

# Import common components
from networks.tf_networks import build_bootstrapped_dqn # Specific network builder
from replay_memory import PER_ReplayBuffer # Can use PER
# Import custom layers needed for loading model
from layers.custom_layers_tf import MeanReducer # Needed if Dueling arch used in base
from networks.tf_networks import StackLayer # Custom layer used in builder
from agents.base_agent import BaseAgent

class BootstrappedDQNAgent(BaseAgent):
    """ Bootstrapped DQN Agent (TF). Uses ensemble of Q-heads for exploration. """
    def __init__(self, game,
                 # Bootstrap specific parameters
                 num_heads=10,
                 mask_probability=0.5, # Probability of updating a head for a given transition
                 # Standard DQN parameters
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 hidden_units=(64, 64), activation='relu',
                 huber_delta=1.0, # Delta for Huber loss
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size

        # Bootstrap parameters
        self.num_heads = max(1, num_heads)
        self.mask_probability = mask_probability
        self.active_head_index = 0 # Index of the head used for acting

        # Store hyperparameters
        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.per_alpha = per_alpha; self.per_beta_start = per_beta_start; self.per_beta = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.huber_delta = huber_delta # Store delta for manual Huber loss

        # Initialize PER buffer
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.per_alpha)
        # frame_idx inherited

        # Build multi-headed networks
        self.model = build_bootstrapped_dqn(
            (self.state_size,), self.action_size, self.num_heads,
            hidden_units, activation, name="BootstrapDQN_Online")
        self.target_model = build_bootstrapped_dqn(
            (self.state_size,), self.action_size, self.num_heads,
            hidden_units, activation, name="BootstrapDQN_Target")
        # Ensure num_heads attribute is on models (builder should add it)
        if not hasattr(self.model, 'num_heads'): self.model.num_heads = self.num_heads
        if not hasattr(self.target_model, 'num_heads'): self.target_model.num_heads = self.num_heads


        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        # Loss calculated manually in _train_step_tf using self.huber_delta

        self.update_target_model() # Initial sync
        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

        # Sample initial acting head
        self.sample_active_head()


    def sample_active_head(self):
        """Samples a head to use for acting in the next episode."""
        self.active_head_index = random.randrange(self.num_heads)
        self._log(logging.DEBUG, f"Sampled active head index: {self.active_head_index}")

    def remember(self, state, action, reward, next_state, done):
        """Generates head mask and stores experience in PER buffer."""
        # Generate a binary mask (0.0 or 1.0) for each head
        head_mask = np.random.binomial(1, self.mask_probability, self.num_heads).astype(np.float32)
        # Store (s, a, r, s', d, mask)
        self.replay_buffer.add((state.astype(np.float32),
                                action,
                                np.float32(reward),
                                next_state.astype(np.float32),
                                done,
                                head_mask))

    def get_action(self, state):
        """Select action using the currently active head."""
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        # Predict Q-values for all heads: shape (1, num_heads, num_actions)
        all_head_q_values = self.model(state_input, training=False).numpy()[0] # Use __call__
        # Select Q-values for the currently active head
        active_head_q_values = all_head_q_values[self.active_head_index] # Shape (num_actions,)
        # Return the action with the highest Q-value for the active head
        return np.argmax(active_head_q_values)

    def update_beta(self): # PER Beta annealing
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.per_beta_frames, 1.0)
        self.per_beta = self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, head_masks, weights):
        """Performs Bootstrapped DQN training step with head masking."""
        # --- Target Calculation (Standard DQN target, calculated per head) ---
        # Get Q-values from target network for next states, all heads
        next_q_target_all_heads = self.target_model(next_states, training=False) # Shape: (b, h, a)
        # Find the max Q-value across actions for each head
        next_q_max_per_head = tf.reduce_max(next_q_target_all_heads, axis=2) # Shape: (b, h)

        # Prepare rewards and dones for broadcasting
        rewards_exp = tf.expand_dims(tf.cast(rewards, tf.float32), 1) # (b, 1)
        dones_exp = tf.expand_dims(tf.cast(dones, tf.float32), 1)    # (b, 1)
        # Calculate target value for each head: T_h = R + gamma * max_a' Q_target_h(s', a') * (1 - done)
        target_per_head = rewards_exp + self.gamma * next_q_max_per_head * (1.0 - dones_exp) # Shape: (b, h)

        # --- Loss Calculation and Gradient Update ---
        with tf.GradientTape() as tape:
            # Get Q-values from online network for current states, all heads
            current_q_all_heads = self.model(states, training=True) # Shape: (b, h, a)

            # Gather Q-values for the specific action taken, for each head
            batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32) # (b,)
            actions_flat = tf.cast(actions, dtype=tf.int32) # (b,)
            # Create meshgrid indices for gather_nd
            b_mesh, h_mesh = tf.meshgrid(batch_indices, tf.range(self.num_heads, dtype=tf.int32), indexing='ij') # (b, h), (b, h)
            # Tile actions to match head dimension
            a_mesh = tf.tile(tf.expand_dims(actions_flat, 1), [1, self.num_heads]) # (b, h)
            # Stack indices: shape (b, h, 3) where each element is [batch_idx, head_idx, action_idx]
            gather_indices_3d = tf.stack([b_mesh, h_mesh, a_mesh], axis=-1)
            # Gather the Q-values for the taken actions
            current_q_action_taken_per_head = tf.gather_nd(current_q_all_heads, gather_indices_3d) # Shape: (b, h)

            # Calculate TD error per head: TD_h = Target_h - Q_online_h(s, a)
            td_error_per_head = target_per_head - current_q_action_taken_per_head # Shape: (b, h)

            # Calculate Huber loss manually per element
            abs_td_error = tf.abs(td_error_per_head)
            quadratic = tf.minimum(abs_td_error, self.huber_delta)
            linear = abs_td_error - quadratic
            # Huber loss = 0.5 * quadratic^2 + delta * linear
            element_loss_per_head = 0.5 * tf.square(quadratic) + self.huber_delta * linear # Shape (b, h)

            # Apply Head Mask (from PER buffer) and Importance Sampling Weights
            # Mask out losses for heads not selected for this transition
            masked_loss_per_head = element_loss_per_head * head_masks # Shape: (b, h)
            # Apply PER IS weights (shape b -> expanded to b, 1)
            weighted_masked_loss = masked_loss_per_head * tf.expand_dims(weights, 1) # Shape: (b, h)

            # Calculate Average Loss per sample (average over active heads)
            # Sum loss across active heads for each sample
            sum_masked_loss_per_sample = tf.reduce_sum(weighted_masked_loss, axis=1) # Shape: (b,)
            # Count number of active heads per sample
            num_active_heads_per_sample = tf.reduce_sum(head_masks, axis=1) # Shape: (b,)
            # Average loss per sample (avoid division by zero)
            average_loss_per_sample = sum_masked_loss_per_sample / tf.maximum(num_active_heads_per_sample, 1e-6)
            # Final loss is the mean over the batch
            loss = tf.reduce_mean(average_loss_per_sample)

        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Calculate TD Errors for PER update (average TD error across active heads)
        # Use the already calculated td_error_per_head
        masked_td_errors = td_error_per_head * head_masks # Shape: (b, h)
        sum_masked_td_errors = tf.reduce_sum(masked_td_errors, axis=1) # Shape: (b,)
        # Average TD error per sample
        average_td_error = sum_masked_td_errors / tf.maximum(num_active_heads_per_sample, 1e-6) # Shape: (b,)

        # Return average TD errors and the scalar loss
        return average_td_error, loss


    def learn(self):
        """Samples batch, performs training step, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()

        # Sample includes head_mask: (s, a, r, s', d, mask)
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.per_beta)
        if batch is None:
            self._log(logging.WARNING, "Bootstrapped sampling failed.")
            return None

        # Prepare Batch Data
        states_np = np.array([b[0] for b in batch], dtype=np.float32)
        actions_np = np.array([b[1] for b in batch], dtype=np.int32)
        rewards_np = np.array([b[2] for b in batch], dtype=np.float32)
        next_states_np = np.array([b[3] for b in batch], dtype=np.float32)
        dones_np = np.array([b[4] for b in batch], dtype=np.float32) # Float for TF
        head_masks_np = np.array([b[5] for b in batch], dtype=np.float32) # (b, num_h)

        # Convert to tensors
        states_tf = tf.convert_to_tensor(states_np); actions_tf = tf.convert_to_tensor(actions_np)
        rewards_tf = tf.convert_to_tensor(rewards_np); next_states_tf = tf.convert_to_tensor(next_states_np)
        dones_tf = tf.convert_to_tensor(dones_np); head_masks_tf = tf.convert_to_tensor(head_masks_np)
        weights_tf = tf.convert_to_tensor(weights_np, dtype=tf.float32)

        # Execute train step
        avg_td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, head_masks_tf, weights_tf)

        # Update PER priorities using the average TD error across active heads
        avg_td_errors_np = avg_td_errors_tf.numpy()
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idxs[i], avg_td_errors_np[i])

        # Return metrics
        metrics = {
            'loss': loss_tf.numpy(),
            'mean_td_error': np.mean(avg_td_errors_np), # Avg TD error used for PER
            'beta': self.per_beta
        }
        return metrics


    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"BootstrappedDQNAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
            # Ensure target model also has num_heads attribute
            if not hasattr(self.target_model, 'num_heads'):
                 self.target_model.num_heads = getattr(self.model, 'num_heads', self.num_heads)

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        # Bootstrapped DQN uses Dense layers + custom StackLayer
        custom_objects = {'StackLayer': StackLayer}
        loaded_model = self._load_keras_model(model_file, custom_objects=custom_objects) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            # Ensure num_heads attribute is set if not saved/loaded correctly
            if not hasattr(self.model, 'num_heads') or self.model.num_heads != self.num_heads:
                self._log(logging.WARNING, f"Loaded model num_heads mismatch or missing. Setting to configured value: {self.num_heads}.")
                self.model.num_heads = self.num_heads
            self.update_target_model()

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/bootstrapped_dqn_agent.py ---
