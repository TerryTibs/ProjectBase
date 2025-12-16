# --- START OF FILE project/agents/rnd_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import random
from collections import deque
import logging # Import logging

# Import common components
from networks.tf_networks import build_dense_dqn, build_rnd_network # RND specific nets
from replay_memory import PER_ReplayBuffer # Can use PER
from agents.base_agent import BaseAgent

class RNDDQNAgent(BaseAgent):
    """ DQN Agent with Random Network Distillation (RND) (TF). """
    def __init__(self, game,
                 # RND specific parameters
                 rnd_output_dim=128,
                 rnd_hidden_units=(64, 64),
                 intrinsic_reward_scale=0.01,
                 normalize_intrinsic=True, # Flag to normalize intrinsic rewards
                 reward_history_len=1000, # Window for running stats
                 # Standard DQN parameters
                 lr=0.0001, # Base LR, RND optimizer LR might be derived
                 gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 dqn_hidden_units=(64, 64), activation='relu',
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        # RND agent assumes vector input for its networks
        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size
        self.input_shape = (self.state_size,)

        # RND parameters
        self.rnd_output_dim = rnd_output_dim
        self.intrinsic_scale = intrinsic_reward_scale
        self.normalize_intrinsic = normalize_intrinsic

        # Store hyperparameters
        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.per_alpha = per_alpha; self.per_beta_start = per_beta_start; self.per_beta = per_beta_start
        self.per_beta_frames = per_beta_frames

        # Initialize PER buffer
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.per_alpha)
        # frame_idx inherited

        # Build DQN Networks
        self.model = build_dense_dqn(self.input_shape, self.action_size, dqn_hidden_units, activation, name="RND_DQN_Online")
        self.target_model = build_dense_dqn(self.input_shape, self.action_size, dqn_hidden_units, activation, name="RND_DQN_Target")

        # Build RND Networks
        self.rnd_target_net = build_rnd_network(self.input_shape, self.rnd_output_dim, rnd_hidden_units, activation, name="RND_Target")
        self.rnd_target_net.trainable = False # Freeze target network weights
        self.rnd_predictor_net = build_rnd_network(self.input_shape, self.rnd_output_dim, rnd_hidden_units, activation, name="RND_Predictor")

        # Optimizers and Losses
        self.dqn_optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.dqn_loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE) # Element-wise Huber
        # RND predictor uses MSE loss
        # Often use lower LR for predictor, e.g., lr/4 or lr/10
        rnd_lr = self.learning_rate / 4.0
        self.rnd_optimizer = optimizers.Adam(learning_rate=rnd_lr, clipnorm=10.0)
        self.rnd_loss_fn = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) # Element-wise MSE

        self.update_target_model() # Sync DQN target model

        # Load models if path exists
        if self.model_path_base:
             self.load(self.model_path_base) # Uses helper which logs

        # Epsilon greedy needed for DQN part
        self.epsilon_start = 1.0; self.epsilon_end = 0.01; self.epsilon_decay = 0.9999
        self.epsilon = self.epsilon_start

        # For normalizing intrinsic rewards - Use deque for running stats
        self.reward_history_len = reward_history_len
        self.intrinsic_reward_history = deque(maxlen=self.reward_history_len)
        self.intrinsic_reward_mean = 0.0
        self.intrinsic_reward_std = 1.0

    def remember(self, state, action, reward, next_state, done):
        """Stores only extrinsic reward. Intrinsic reward calculated during learning."""
        self.replay_buffer.add((state.astype(np.float32),
                                action,
                                np.float32(reward),
                                next_state.astype(np.float32),
                                done))

    def get_action(self, state):
        """Epsilon-greedy action selection using the DQN model."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        # Exploit
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        q_values = self.model(state_input, training=False).numpy() # Use __call__
        return np.argmax(q_values[0])

    def update_beta(self): # PER Beta annealing
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.per_beta_frames, 1.0)
        self.per_beta = self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    def _update_reward_stats(self):
        """Updates running mean and std dev of intrinsic rewards from history."""
        if len(self.intrinsic_reward_history) > 1:
            rewards_np = np.array(self.intrinsic_reward_history)
            self.intrinsic_reward_mean = np.mean(rewards_np)
            self.intrinsic_reward_std = np.std(rewards_np) + 1e-8 # Add epsilon for stability
        else:
            # Not enough history, keep defaults
            self.intrinsic_reward_mean = 0.0
            self.intrinsic_reward_std = 1.0
        self._log(logging.DEBUG, f"Updated RND stats: Mean={self.intrinsic_reward_mean:.4f}, Std={self.intrinsic_reward_std:.4f}")


    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs RND DQN training step."""
        # --- RND Intrinsic Reward Calculation & Predictor Update ---
        # Target features are fixed (no gradient tracking needed)
        target_features = self.rnd_target_net(next_states, training=False)

        with tf.GradientTape() as rnd_tape:
            # Predict features using the predictor network (track gradients)
            predicted_features = self.rnd_predictor_net(next_states, training=True)
            # Element-wise MSE loss - this is the raw intrinsic reward signal
            intrinsic_rewards_raw = self.rnd_loss_fn(tf.stop_gradient(target_features), predicted_features) # Shape (batch_size,)
            # RND predictor aims to minimize the mean MSE over the batch
            rnd_loss = tf.reduce_mean(intrinsic_rewards_raw)

        # Apply gradients to RND predictor network
        rnd_grads = rnd_tape.gradient(rnd_loss, self.rnd_predictor_net.trainable_variables)
        self.rnd_optimizer.apply_gradients(zip(rnd_grads, self.rnd_predictor_net.trainable_variables))

        # --- Normalize Intrinsic Reward (using current running stats) ---
        # Pass mean and std deviation as constants to the tf.function
        intr_mean_tf = tf.constant(self.intrinsic_reward_mean, dtype=tf.float32)
        intr_std_tf = tf.constant(self.intrinsic_reward_std, dtype=tf.float32)

        if self.normalize_intrinsic:
            normalized_intrinsic_rewards = (intrinsic_rewards_raw - intr_mean_tf) / intr_std_tf
        else:
            normalized_intrinsic_rewards = intrinsic_rewards_raw

        # --- Combine with extrinsic reward ---
        # Use stop_gradient as the intrinsic reward contributes to the DQN target but shouldn't affect DQN gradients directly via this path
        combined_rewards = tf.cast(rewards, tf.float32) + self.intrinsic_scale * tf.stop_gradient(normalized_intrinsic_rewards)

        # --- Standard DDQN target calculation using combined reward ---
        # Use online network for action selection on next state
        online_next_q = self.model(next_states, training=False)
        next_actions = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        # Use target network for value estimation of selected action
        target_next_q = self.target_model(next_states, training=False)
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)
        # Calculate DQN target: Combined_R + gamma * Q_target(s', best_action) * (1 - done)
        target = combined_rewards + self.gamma * ddqn_next_val * (1.0 - tf.cast(dones, tf.float32))

        # --- DQN Loss Calculation and Gradient Update ---
        with tf.GradientTape() as dqn_tape:
            # Get Q-values from online DQN model for the actions taken
            current_q_all = self.model(states, training=True)
            action_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            # Calculate element-wise Huber loss
            element_loss = self.dqn_loss_fn(tf.stop_gradient(target), current_q)
            # Apply PER Importance Sampling weights
            weighted_element_loss = tf.cast(weights, tf.float32) * element_loss
            # Reduce to scalar loss
            dqn_loss = tf.reduce_mean(weighted_element_loss)

        # Apply gradients to DQN online model
        dqn_grads = dqn_tape.gradient(dqn_loss, self.model.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(dqn_grads, self.model.trainable_variables))

        # TD Errors for PER update (based on combined reward target)
        td_errors = tf.abs(target - current_q)

        # Return TD errors, DQN loss, RND loss, and raw intrinsic rewards
        return td_errors, dqn_loss, rnd_loss, intrinsic_rewards_raw

    def learn(self):
        """Samples batch, performs RND+DQN training, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()

        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.per_beta)
        if batch is None:
            self._log(logging.WARNING,"RND DQN sampling failed.")
            return None

        # Prepare batch data
        states_np=np.array([b[0] for b in batch],dtype=np.float32); actions_np=np.array([b[1] for b in batch],dtype=np.int32)
        rewards_np=np.array([b[2] for b in batch],dtype=np.float32); next_states_np=np.array([b[3] for b in batch],dtype=np.float32)
        dones_np=np.array([b[4] for b in batch],dtype=np.float32) # Float for TF

        # Convert to tensors
        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np); weights_tf=tf.convert_to_tensor(weights_np,dtype=tf.float32)

        # Perform training step
        td_errors_tf, dqn_loss_tf, rnd_loss_tf, intrinsic_rewards_tf = self._train_step_tf(
            states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf
        )

        # Update PER priorities
        td_errors_np = td_errors_tf.numpy()
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idxs[i], td_errors_np[i])

        # Update intrinsic reward running statistics using the raw rewards
        self.intrinsic_reward_history.extend(intrinsic_rewards_tf.numpy())
        self._update_reward_stats() # Recalculate mean/std based on updated history

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # Return metrics
        metrics = {
            'dqn_loss': dqn_loss_tf.numpy(),
            'rnd_loss': rnd_loss_tf.numpy(),
            'mean_td_error': np.mean(td_errors_np),
            'beta': self.per_beta,
            'epsilon': self.epsilon,
            'intrinsic_reward_mean': self.intrinsic_reward_mean,
            'intrinsic_reward_std': self.intrinsic_reward_std
        }
        return metrics


    def update_target_model(self):
        """Copies weights for DQN target model."""
        self._log(logging.INFO, f"RNDDQNAgent updating DQN target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
        # RND target net is fixed, no update needed

    def load(self, name_base):
        """Loads DQN and RND predictor models."""
        dqn_model_file = name_base + "_dqn.keras"
        rnd_pred_file = name_base + "_rnd_predictor.keras"
        # Load DQN model
        loaded_dqn_model = self._load_keras_model(dqn_model_file) # Helper logs
        if loaded_dqn_model:
            self.model = loaded_dqn_model
            self.update_target_model() # Sync DQN target
        # Load RND predictor model
        loaded_rnd_pred = self._load_keras_model(rnd_pred_file) # Helper logs
        if loaded_rnd_pred:
            self.rnd_predictor_net = loaded_rnd_pred
        # RND target network weights are randomly initialized and fixed, not loaded

    def save(self, name_base):
        """Saves DQN and RND predictor models."""
        dqn_model_file = name_base + "_dqn.keras"
        rnd_pred_file = name_base + "_rnd_predictor.keras"
        # Save DQN model
        self._save_keras_model(self.model, dqn_model_file) # Helper logs
        # Save RND predictor model
        self._save_keras_model(self.rnd_predictor_net, rnd_pred_file) # Helper logs
        # RND target model is not saved

# --- END OF FILE project/agents/rnd_dqn_agent.py ---
