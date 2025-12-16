# --- START OF FILE project/agents/dqn_agents/noisy_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import random
import logging
from collections import deque # Use standard deque

# Import common components
from networks.tf_networks import build_dueling_dqn # Or build_dense_dqn
# ** Corrected import path for custom layers/base agent **
from layers.custom_layers_tf import NoisyLinear, MeanReducer # Needed for loading
from agents.base_agent import BaseAgent

class NoisyDQNAgent(BaseAgent):
    """ DQN Agent using Noisy Nets for exploration and standard replay buffer. """
    def __init__(self, game,
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # Hard updates
                 noisy_std_init=0.5,
                 # Network params (using Dueling structure)
                 shared_hidden_units=(128, 128), activation='relu',
                 model_path=None,
                 logger=None):
        # ** Pass logger to super **
        super().__init__(game, model_path, logger=logger)

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size

        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq

        # Use standard deque replay buffer
        self.replay_buffer = deque(maxlen=self.memory_size)
        # frame_idx inherited

        # Build Networks (Dueling with Noisy Layers)
        self.model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=True, noisy_std_init=noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="NoisyDQN_Online")
        self.target_model = build_dueling_dqn(
            (self.state_size,), self.action_size, noisy=True, noisy_std_init=noisy_std_init,
            shared_hidden_units=shared_hidden_units, activation=activation, name="NoisyDQN_Target")

        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        # ** Use standard Huber loss, reduced to mean over batch **
        self.loss_fn = losses.Huber(reduction='sum_over_batch_size') # Corrected reduction

        self.update_target_model() # Initial hard sync

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

        # No epsilon needed due to Noisy Nets

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the deque."""
        self.replay_buffer.append((state.astype(np.float32), action, np.float32(reward),
                                   next_state.astype(np.float32), done))

    def get_action(self, state):
        """Returns action based on noisy network prediction."""
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        q_values = self.model(state_input, training=True).numpy() # Use noise
        return np.argmax(q_values[0])

    # No update_beta needed

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones):
        """Performs the core training calculations (DDQN loss, no PER)."""
        # Double DQN Target Calculation
        online_next_q = self.model(next_states, training=True) # Use noise for action selection
        next_actions = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        target_next_q = self.target_model(next_states, training=False) # Target without noise

        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions], axis=1)
        ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)
        target = tf.cast(rewards, tf.float32) + self.gamma * ddqn_next_val * (1.0 - tf.cast(dones, tf.float32))

        # Loss Calculation and Gradients
        with tf.GradientTape() as tape:
            current_q_all = self.model(states, training=True) # Online model needs noise
            action_indices = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices)
            # Calculate loss (reduction handled by loss_fn instance)
            loss = self.loss_fn(target, current_q) # Loss instance handles reduction

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Return scalar loss
        return loss

    def learn(self):
        """Samples batch uniformly, performs training step, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None

        # Uniform random sampling
        mini_batch = random.sample(self.replay_buffer, self.batch_size)

        states_np=np.array([t[0] for t in mini_batch],dtype=np.float32); actions_np=np.array([t[1] for t in mini_batch],dtype=np.int32)
        rewards_np=np.array([t[2] for t in mini_batch],dtype=np.float32); next_states_np=np.array([t[3] for t in mini_batch],dtype=np.float32)
        dones_np=np.array([t[4] for t in mini_batch],dtype=np.float32) # Float for TF

        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np)

        loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf)

        # Periodic hard target update is handled by the main training loop calling update_target_model

        metrics = {'loss': loss_tf.numpy()}
        return metrics

    def update_target_model(self):
        """Copies weights to target model (Hard Update)."""
        self._log(logging.INFO, f"NoisyDQNAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        custom_objects = {'NoisyLinear': NoisyLinear, 'MeanReducer': MeanReducer}
        loaded_model = self._load_keras_model(model_file, custom_objects) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self.update_target_model() # Sync target after loading

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/dqn_agents/noisy_dqn_agent.py ---
