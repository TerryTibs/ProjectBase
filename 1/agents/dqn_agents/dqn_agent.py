import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import os
import logging # Import logging if you use self._log

# Import network builder and base agent
# Assuming networks and agents folders are peers within the project directory
try:
    from networks.tf_networks import build_dense_dqn
    from agents.base_agent import BaseAgent # Using BaseAgent
except ImportError:
    # Adjust path if running script directly from dqn_agents folder (less common)
    from ...networks.tf_networks import build_dense_dqn
    from ..base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """ Basic DQN Agent (TF) - Uses network builder """
    def __init__(self, game,
                 learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 memory_size=10000, batch_size=64, target_update_freq=1000,
                 hidden_units=(64, 64), activation='relu',
                 model_path=None,
                 logger=None):
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger)

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game object must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size
        self.action_size = game.action_size

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replay_buffer = deque(maxlen=self.memory_size)
        self.target_update_freq = target_update_freq

        # Build Networks
        self.model = build_dense_dqn(
            (self.state_size,), self.action_size, hidden_units, activation, "DQNAgent_Online")
        self.target_model = build_dense_dqn(
            (self.state_size,), self.action_size, hidden_units, activation, "DQNAgent_Target")

        # Compile Online Model Initially
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        self.update_target_model() # Initial sync

        if self.model_path_base:
            self.load(self.model_path_base)

    def remember(self, state, action, reward, next_state, done):
        """Stores experience tuple in the replay buffer (deque)."""
        # Ensure states are float32 for the network and reward is float
        self.replay_buffer.append((state.astype(np.float32),
                                   action,
                                   np.float32(reward),
                                   next_state.astype(np.float32),
                                   done))

    def get_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
            q_values = self.model(state_input, training=False).numpy()
            return np.argmax(q_values[0])


    def learn(self):
        # Trains the network using a batch from the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough experiences yet

        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([t[0] for t in mini_batch], dtype=np.float32)
        actions = np.array([t[1] for t in mini_batch])
        rewards = np.array([t[2] for t in mini_batch])
        next_states = np.array([t[3] for t in mini_batch], dtype=np.float32)
        dones = np.array([t[4] for t in mini_batch])

        # Predict Q-values for current states and next states
        current_q_values_batch = self.model(states, training=False).numpy()
        # Use target network for next state Q-values (stability)
        next_q_values_batch = self.target_model(next_states, training=False).numpy()

        # Create target batch, starting with current predictions
        targets_batch = np.copy(current_q_values_batch)

        # Calculate target Q-values using Bellman equation
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                # Standard DQN target: R + gamma * max_a' Q_target(s', a')
                target += self.gamma * np.amax(next_q_values_batch[i])
            # Update the Q-value for the action actually taken
            targets_batch[i][actions[i]] = target

        # Train the online model using the calculated targets
        # Use train_on_batch for simpler DQN update with compiled model
        loss = self.model.train_on_batch(states, targets_batch)
        # train_on_batch returns the loss directly in TF 2.x
        loss_value = loss if isinstance(loss, (float, np.number)) else None

        # Decay epsilon after learning step
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Return metrics
        metrics = {'loss': loss_value}
        return metrics


    def update_target_model(self):
        """Copies weights from the online model to the target model."""
        self._log(logging.INFO, f"DQNAgent updating target model at frame {self.frame_idx}") # Use self._log
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        # Loads model weights and recompiles the model
        model_file = name_base + ".keras"
        loaded_model = self._load_keras_model(model_file) # Helper handles logging
        if loaded_model:
            self.model = loaded_model
            self._log(logging.INFO, f"{type(self).__name__}: Re-compiling loaded model...") # Use self._log
            # Re-compile the model after loading weights to set optimizer and loss
            self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            self._log(logging.INFO, f"{type(self).__name__}: Model re-compiled.") # Use self._log
            self.update_target_model() # Sync target model weights

    def save(self, name_base):
        # Saves model weights using the helper
        model_file = name_base + ".keras"
        # Saving with include_optimizer=True (default for .keras) is fine here
        # as we re-compile on load anyway.
        self._save_keras_model(self.model, model_file) # Use helper
