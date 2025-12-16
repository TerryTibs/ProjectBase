# --- START OF FILE project/agents/cnn_dqn_agent_tf.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import logging # Import logging

# Import network builder, PER buffer, custom layers, and base agent
from networks.tf_networks import build_cnn_dqn
from replay_memory import PER_ReplayBuffer
from layers.custom_layers_tf import NoisyLinear # Needed for loading if noisy_fc=True
from agents.base_agent import BaseAgent

class CNN_DQN_Agent_TF(BaseAgent):
    """
    CNN-DQN Agent (TF) - Uses network builder.
    Can handle 3D pixel (uint8) or 2D grid (float32) states.
    """
    def __init__(self, game,
                 lr=0.0001, gamma=0.99, batch_size=32,
                 memory_size=100000, target_update_freq=10000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 noisy_fc=True, noisy_std_init=0.5, # Network options
                 fc_units=512, activation='relu', # Network options
                 model_path=None,
                 input_type='screen', # 'screen' (3D uint8) or 'grid' (2D float32)
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        self.input_type = input_type
        # Determine input shape and dtype based on input_type
        if self.input_type == 'screen':
            if not hasattr(game, 'get_screen_size') or not callable(game.get_screen_size):
                raise ValueError("Game needs 'get_screen_size' for 'screen' input type.")
            self.input_shape = game.get_screen_size() # (H, W, C)
            self.input_dtype = tf.uint8
            if len(self.input_shape) != 3: raise ValueError(f"Expected 3D shape (H, W, C) for screen input, got {self.input_shape}")
            self._log(logging.INFO, f"{type(self).__name__}: Using SCREEN input, shape={self.input_shape}, dtype={self.input_dtype}")
        elif self.input_type == 'grid':
            if not hasattr(game, 'get_grid_shape') or not callable(game.get_grid_shape):
                raise ValueError("Game needs 'get_grid_shape' for 'grid' input type.")
            self.input_shape = game.get_grid_shape() # (Rows, Cols)
            self.input_dtype = tf.float32
            if len(self.input_shape) != 2: raise ValueError(f"Expected 2D shape (Rows, Cols) for grid input, got {self.input_shape}")
            self._log(logging.INFO, f"{type(self).__name__}: Using GRID input, shape={self.input_shape}, dtype={self.input_dtype}")
        else:
            raise ValueError(f"Invalid input_type '{self.input_type}'. Must be 'screen' or 'grid'.")

        if not hasattr(game, 'get_action_space') or not callable(game.get_action_space):
             raise ValueError("Game needs 'get_action_space'.")
        self.action_size = game.get_action_space()

        # Store hyperparameters
        self.gamma = gamma; self.batch_size = batch_size; self.target_update_freq = target_update_freq
        self.alpha = per_alpha; self.beta_start = per_beta_start; self.beta = per_beta_start
        self.beta_frames = per_beta_frames; self.noisy_fc = noisy_fc # Store if noisy FC used

        # Initialize PER buffer
        self.replay_buffer = PER_ReplayBuffer(memory_size, alpha=self.alpha)
        # frame_idx inherited

        # Build Networks - PASSING input_dtype to the builder
        self.model = build_cnn_dqn(
            input_shape=self.input_shape, num_actions=self.action_size, noisy_fc=noisy_fc,
            noisy_std_init=noisy_std_init, fc_units=fc_units, activation=activation, name="CNN_DQN_TF_Online",
            input_dtype=self.input_dtype) # Pass dtype here
        self.target_model = build_cnn_dqn(
            input_shape=self.input_shape, num_actions=self.action_size, noisy_fc=noisy_fc,
            noisy_std_init=noisy_std_init, fc_units=fc_units, activation=activation, name="CNN_DQN_TF_Target",
            input_dtype=self.input_dtype) # Pass dtype here

        # Optimizer and Loss
        self.loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.optimizer = optimizers.Adam(learning_rate=lr, clipnorm=10.0) # Use passed lr
        self.update_target_model() # Initial sync

        # Epsilon not needed if noisy_fc=True
        self.epsilon = 0.0 # Assume noisy nets for CNN

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

    def remember(self, state, action, reward, next_state, done):
        """Stores state (uint8 for screen, float32 for grid) in the PER buffer."""
        # Ensure state has the correct dtype before adding
        expected_dtype = np.uint8 if self.input_dtype == tf.uint8 else np.float32
        self.replay_buffer.add((state.astype(expected_dtype),
                                action,
                                np.float32(reward), # Ensure reward is float
                                next_state.astype(expected_dtype),
                                done))

    def get_action(self, state):
        """Selects action based on CNN output (using noise if enabled)."""
        # Convert state to tensor with the expected dtype for the model's Input layer
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=self.input_dtype)
        # Pass training=True only if using NoisyLinear layers
        q_values = self.model(state_tensor, training=self.noisy_fc).numpy()
        return np.argmax(q_values[0])

    def update_beta(self):
         """Anneals beta linearly."""
         fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
         self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """
        Performs a single training step (DDQN, PER loss) for CNN.
        Input tensors (states, next_states) will have the dtype defined by self.input_dtype.
        """
        # Double DQN Target Calculation
        online_next_q = self.model(next_states, training=self.noisy_fc)
        next_actions = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        target_next_q = self.target_model(next_states, training=False)

        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)
        target = tf.cast(rewards, tf.float32) + self.gamma * ddqn_next_val * (1.0 - tf.cast(dones, tf.float32))

        # Loss Calculation and Gradients
        with tf.GradientTape() as tape:
            current_q_all = self.model(states, training=self.noisy_fc)
            action_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            element_loss = self.loss_fn(target, current_q)
            weighted_loss = tf.cast(weights, tf.float32) * element_loss
            loss = tf.reduce_mean(weighted_loss) # Mean over batch

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        td_errors = tf.abs(target - current_q)
        return td_errors, loss # Return loss as well

    def learn(self):
        """Samples batch, performs training step, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()

        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None:
            self._log(logging.WARNING, "TF CNN sampling failed.")
            return None

        # Determine expected dtype from model config
        expected_dtype_np = np.uint8 if self.input_dtype == tf.uint8 else np.float32

        # Prepare batch data
        states_np = np.array([b[0] for b in batch], dtype=expected_dtype_np)
        actions_np = np.array([b[1] for b in batch], dtype=np.int32)
        rewards_np = np.array([b[2] for b in batch], dtype=np.float32)
        next_states_np = np.array([b[3] for b in batch], dtype=expected_dtype_np)
        dones_np = np.array([b[4] for b in batch], dtype=np.float32) # Float for TF

        # Convert to tensors
        states_tf = tf.convert_to_tensor(states_np)
        actions_tf = tf.convert_to_tensor(actions_np)
        rewards_tf = tf.convert_to_tensor(rewards_np)
        next_states_tf = tf.convert_to_tensor(next_states_np)
        dones_tf = tf.convert_to_tensor(dones_np)
        weights_tf = tf.convert_to_tensor(weights_np, dtype=tf.float32)

        # Perform train step
        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf)

        # Update priorities
        td_errors_np = td_errors_tf.numpy()
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idxs[i], td_errors_np[i])

        # Return metrics
        metrics = {
            'loss': loss_tf.numpy(),
            'mean_td_error': np.mean(td_errors_np),
            'beta': self.beta
        }
        return metrics

    def update_target_model(self):
        """Copy weights from the online model to the target model."""
        self._log(logging.INFO, f"CNN_DQN_Agent_TF updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        # Include NoisyLinear only if the model architecture uses it
        custom_objects = {'NoisyLinear': NoisyLinear} if self.noisy_fc else {}
        loaded_model = self._load_keras_model(model_file, custom_objects) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self.update_target_model()

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/cnn_dqn_agent_tf.py ---
