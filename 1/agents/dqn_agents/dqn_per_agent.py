# --- START OF FILE project/agents/dqn_per_agent.py ---

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os
import logging # Import logging

# Import network builder, PER buffer, and base agent
from networks.tf_networks import build_per_dqn # Or build_dense_dqn
from replay_memory import PER_ReplayBuffer
from agents.base_agent import BaseAgent

class DQN_PER_Agent(BaseAgent):
    """ DQN Agent with PER (TF) - Uses network builder """
    def __init__(self, game,
                 learning_rate=0.0001, gamma=0.99, batch_size=64,
                 memory_size=50000, target_update_freq=1000, # Target update based on frames now
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999, # Applied per learn step
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 hidden_units=(64, 32, 16), activation='relu', # Network params
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size
        self.action_size = game.action_size

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay # Applied per learn step
        self.learning_rate = learning_rate # Store for re-compile
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq # In frames

        # PER Parameters
        self.alpha = per_alpha
        self.beta_start = per_beta_start
        self.beta = per_beta_start # Current beta
        self.beta_frames = per_beta_frames

        # Use centralized PER Replay Buffer
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.alpha)
        # frame_idx inherited from BaseAgent

        # Build Networks
        self.model = build_per_dqn( # Or build_dense_dqn
            (self.state_size,), self.action_size, hidden_units, activation, "DQN_PER_Online")
        self.target_model = build_per_dqn( # Or build_dense_dqn
            (self.state_size,), self.action_size, hidden_units, activation, "DQN_PER_Target")

        # Compile Online Model Initially
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        self.update_target_model()

        if self.model_path_base:
            self.load(self.model_path_base) # load calls _load_keras_model which logs

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the PER buffer."""
        # No need to calculate initial priority here, PER_ReplayBuffer.add handles it
        self.replay_buffer.add((state.astype(np.float32),
                                action,
                                np.float32(reward),
                                next_state.astype(np.float32),
                                done))

    def get_action(self, state):
        """Returns action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        # Exploit
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        q_values = self.model(state_input, training=False).numpy()
        return np.argmax(q_values[0])

    def update_beta(self):
        """Anneals beta based on frame index."""
        fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    def learn(self):
        """Trains the network using PER and returns metrics."""
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough experiences

        self.update_beta() # Anneal beta before sampling

        # Sample Batch using PER
        batch, idxs, weights = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None: # Sampling failed
            self._log(logging.WARNING,"PER sampling failed in DQN_PER_Agent. Skipping learn step.")
            return None

        # Prepare Batch Data
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int32) # Ensure int32 for indexing
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch]) # Boolean is fine here

        # Predict Q-values - use __call__
        current_q_values_batch = self.model(states, training=False).numpy()
        next_q_values_batch = self.target_model(next_states, training=False).numpy()
        targets_batch = np.copy(current_q_values_batch) # Start with current predictions
        td_errors = np.zeros(self.batch_size) # To store errors for priority update

        # Calculate Targets and TD Errors
        for i in range(self.batch_size):
            current_q = current_q_values_batch[i][actions[i]] # Q-value of action taken
            target = rewards[i]
            if not dones[i]:
                # Standard DQN target: R + gamma * max_a' Q_target(s', a')
                target += self.gamma * np.max(next_q_values_batch[i])
            targets_batch[i][actions[i]] = target # Set target for the action taken
            td_errors[i] = abs(target - current_q) # Calculate TD error

        # Update Priorities in SumTree
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idx, td_errors[i])

        # Train using sample_weight for IS correction
        # train_on_batch returns the loss
        loss = self.model.train_on_batch(states, targets_batch, sample_weight=weights)

        # Decay Epsilon (per learn step)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Return metrics
        metrics = {
            'loss': loss,
            'mean_td_error': np.mean(td_errors),
            'beta': self.beta # Include current beta
        }
        return metrics

    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"DQN_PER_Agent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        """Loads model weights and RE-COMPILES the loaded model."""
        model_file = name_base + ".keras"
        loaded_model = self._load_keras_model(model_file) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self._log(logging.INFO, f"{type(self).__name__}: Re-compiling loaded model...")
            # Use the stored learning rate for consistency
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            self._log(logging.INFO, f"{type(self).__name__}: Model re-compiled.")
            self.update_target_model()

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/dqn_per_agent.py ---
