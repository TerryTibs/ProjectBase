# --- START OF FILE project/agents/dqn_agents/averaged_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os
import logging
from collections import deque # Use simple deque

# Import network builder and base agent
from networks.tf_networks import build_dense_dqn
# ** Corrected import path for BaseAgent **
from agents.base_agent import BaseAgent

class AveragedDQNAgent(BaseAgent):
    """ DQN Agent using averaged online network weights for targets. """
    def __init__(self, game,
                 learning_rate=0.001, gamma=0.99,
                 avg_update_tau=0.005, # Tau for Polyak averaging
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 memory_size=50000, batch_size=64,
                 hidden_units=(64, 64), activation='relu',
                 model_path=None,
                 logger=None):
        # ** Pass logger to super **
        super().__init__(game, model_path, logger=logger)

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game object must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size
        self.input_shape = (self.state_size,)

        self.learning_rate = learning_rate; self.gamma = gamma
        self.tau = avg_update_tau # Averaging factor
        self.epsilon = epsilon_start; self.epsilon_min = epsilon_end; self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size; self.memory_size = memory_size

        # Use standard deque replay buffer
        self.replay_buffer = deque(maxlen=self.memory_size)
        # frame_idx inherited

        # Build Online Network
        self.model = build_dense_dqn(self.input_shape, self.action_size, hidden_units, activation, "AvgDQN_Online")
        # Build Averaged Network (same architecture, non-trainable usually, weights managed manually)
        self.avg_model = build_dense_dqn(self.input_shape, self.action_size, hidden_units, activation, "AvgDQN_Average")
        self.avg_model.trainable = False # Weights updated via averaging, not training

        # Optimizer and Loss
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE) # Element-wise

        # Initialize avg_model weights to match online model
        self._sync_average_model(hard=True) # Initial hard sync

        if self.model_path_base:
            self.load(self.model_path_base) # Loads online, avg is synced after load

    def _sync_average_model(self, hard=False):
        """Updates the averaged model weights using Polyak averaging or hard copy."""
        if self.model is None or self.avg_model is None: return

        online_weights = self.model.get_weights()
        if hard:
            self.avg_model.set_weights(online_weights)
            self._log(logging.DEBUG, "Hard sync of average model complete.")
        else:
            avg_weights = self.avg_model.get_weights()
            new_avg_weights = []
            if len(online_weights) != len(avg_weights):
                self._log(logging.ERROR, "Online and average model weight counts differ. Skipping average update.")
                return
            try:
                for online_w, avg_w in zip(online_weights, avg_weights):
                    new_w = self.tau * online_w + (1.0 - self.tau) * avg_w
                    new_avg_weights.append(new_w)
                self.avg_model.set_weights(new_avg_weights)
                self._log(logging.DEBUG, f"Average model updated (tau={self.tau}).")
            except Exception as e:
                self._log(logging.ERROR, f"Error during average model update: {e}", exc_info=True)

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in deque."""
        self.replay_buffer.append((state.astype(np.float32), action, np.float32(reward),
                                   next_state.astype(np.float32), done))

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        q_values = self.model(state_input, training=False).numpy()
        return np.argmax(q_values[0])

    # --- THIS METHOD WAS MISSING ---
    def update_target_model(self):
        """
        Implementation of the abstract method from BaseAgent.
        AveragedDQN does not use a traditional target model,
        so this method does nothing. The averaged model is updated
        during the learn step via _sync_average_model.
        """
        self._log(logging.DEBUG, f"{type(self).__name__}: update_target_model called, but not applicable.")
        pass
    # --- END OF MISSING METHOD ---

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones):
        """Performs DQN training step using the averaged model for targets."""
        # Target Calculation using Averaged Model
        next_q_values_avg = self.avg_model(next_states, training=False) # Explicitly training=False
        max_next_q_avg = tf.reduce_max(next_q_values_avg, axis=1)
        target = tf.cast(rewards, tf.float32) + self.gamma * max_next_q_avg * (1.0 - tf.cast(dones, tf.float32))

        # Loss Calculation and Gradients for Online Model
        with tf.GradientTape() as tape:
            current_q_all = self.model(states, training=True) # Online model is trained
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            # Use element-wise loss then reduce mean
            element_loss = self.loss_fn(tf.stop_gradient(target), current_q)
            loss = tf.reduce_mean(element_loss) # Mean loss over batch

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def learn(self):
        """Samples batch, performs training step, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None

        mini_batch = random.sample(self.replay_buffer, self.batch_size)

        states_np=np.array([t[0] for t in mini_batch],dtype=np.float32); actions_np=np.array([t[1] for t in mini_batch],dtype=np.int32)
        rewards_np=np.array([t[2] for t in mini_batch],dtype=np.float32); next_states_np=np.array([t[3] for t in mini_batch],dtype=np.float32)
        dones_np=np.array([t[4] for t in mini_batch],dtype=np.float32) # Float for TF

        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np)

        loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf)

        # Update the averaged model weights AFTER the online model step
        self._sync_average_model(hard=False) # Soft update

        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

        metrics = {'loss': loss_tf.numpy(), 'epsilon': self.epsilon}
        return metrics

    def load(self, name_base):
        """Loads online model weights and syncs average model."""
        model_file = name_base + "_online.keras" # Save online model only
        loaded_model = self._load_keras_model(model_file)
        if loaded_model:
            self.model = loaded_model
            # After loading, hard-sync the average model to match
            self._sync_average_model(hard=True)
            self._log(logging.INFO, "Synced average model after loading online model.")

    def save(self, name_base):
        """Saves only the online model weights."""
        model_file = name_base + "_online.keras"
        self._save_keras_model(self.model, model_file) # Only save the trained online model

# --- END OF FILE project/agents/dqn_agents/averaged_dqn_agent.py ---
