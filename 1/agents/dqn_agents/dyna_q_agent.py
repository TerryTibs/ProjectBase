# --- START OF FILE project/agents/dyna_q_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import random
import os
from collections import deque
import logging # <<< ADD THIS IMPORT

# Import common components
from networks.tf_networks import build_dense_dqn, build_world_model_network # Assuming 1D state model
from agents.base_agent import BaseAgent
from replay_memory import PER_ReplayBuffer # Can use PER or simple buffer (using deque here)

class DynaQAgent(BaseAgent):
    """
    Dyna-Q Agent (TF). Combines DQN with a learned world model for planning.
    This version assumes a 1D vector state and learns a model predicting 1D next state.
    """
    def __init__(self, game,
                 input_type='vector', # MUST be 'vector' for this simple world model
                 # DQN parameters
                 lr_dqn=0.0005, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 batch_size=64, target_update_freq=1000, # Frames
                 dqn_hidden_units=(64, 64),
                 # World Model parameters
                 lr_model=0.001,
                 model_hidden_units=(128, 128),
                 # Dyna-Q parameters
                 planning_steps=10, # Number of planning updates per real step
                 # Memory
                 memory_size=50000,
                 # Other
                 model_path=None,
                 logger=None): # Accept logger

        super().__init__(game, model_path, logger=logger) # Pass logger

        if input_type != 'vector':
            # Log error before raising
            self._log(logging.ERROR, "This simple DynaQAgent currently only supports 'vector' input type.")
            raise ValueError("This simple DynaQAgent currently only supports 'vector' input type.")
        self.input_type = input_type

        if not hasattr(game, 'get_state_size') or not callable(game.get_state_size):
            raise ValueError("DynaQAgent requires game.get_state_size()")
        self.state_size = game.get_state_size()
        self.action_size = game.get_action_space()
        self.input_shape = (self.state_size,) # For vector input

        # DQN parameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Dyna-Q parameters
        self.planning_steps = planning_steps

        # Build DQN Networks
        self.model = build_dense_dqn(self.input_shape, self.action_size, dqn_hidden_units, 'relu', "DynaQ_DQN_Online")
        self.target_model = build_dense_dqn(self.input_shape, self.action_size, dqn_hidden_units, 'relu', "DynaQ_DQN_Target")
        self.dqn_optimizer = optimizers.Adam(learning_rate=lr_dqn)
        # Instantiate DQN loss object
        self.dqn_loss_fn = losses.Huber()

        # Build World Model Network
        self.world_model = build_world_model_network(self.state_size, model_hidden_units, 'relu', "DynaQ_WorldModel")
        self.model_optimizer = optimizers.Adam(learning_rate=lr_model)
        # Instantiate world model loss objects
        self.model_state_loss_fn = losses.MeanSquaredError()
        self.model_reward_loss_fn = losses.MeanSquaredError()

        # Experience Buffer (simple deque)
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size # Store max size

        # Sync target net and load models
        self.update_target_model()
        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs


    def remember(self, state, action, reward, next_state, done):
        """Stores real experience."""
        if self.input_type != 'vector':
             self._log(logging.ERROR, "DynaQAgent received non-vector state during remember, unexpected.")
             return # Avoid storing wrong state type
        self.memory.append((state.astype(np.float32), action, np.float32(reward), next_state.astype(np.float32), done))

    def get_action(self, state):
        """ Epsilon-greedy action selection based on the Q-model. """
        if self.input_type != 'vector':
             self._log(logging.ERROR, "DynaQAgent get_action called with non-vector state.")
             return random.randrange(self.action_size)
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        else:
            state_input = np.expand_dims(state.astype(np.float32), axis=0)
            q_values = self.model(state_input, training=False).numpy()
            return np.argmax(q_values[0])

    @tf.function
    def _train_q_step(self, states, actions, rewards, next_states, dones):
        """ Performs a single training step for the Q-network (DQN update). """
        # (Logic remains the same)
        next_q_values = self.target_model(next_states, training=False)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        target_q = rewards + self.gamma * max_next_q * (1.0 - tf.cast(dones, tf.float32))
        with tf.GradientTape() as tape:
            q_values_all = self.model(states, training=True)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            q_values_taken = tf.gather_nd(q_values_all, action_indices)
            loss = self.dqn_loss_fn(tf.stop_gradient(target_q), q_values_taken) # Use instantiated loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Optional: Clip gradients
        # grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.dqn_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return tf.reduce_mean(loss) # Return mean loss

    @tf.function
    def _train_model_step(self, states, actions, rewards, next_states):
        """ Performs a single training step for the world model. """
        # TODO: If model uses action, need to prepare action input (e.g., one-hot)
        with tf.GradientTape() as tape:
            # Assuming model takes only state for now
            predicted_next_state, predicted_reward = self.world_model(states, training=True)
            predicted_reward = tf.squeeze(predicted_reward, axis=-1) # Ensure correct shape (b,) if output is (b,1)

            state_loss = self.model_state_loss_fn(next_states, predicted_next_state) # MSE
            reward_loss = self.model_reward_loss_fn(rewards, predicted_reward) # MSE
            total_loss = tf.reduce_mean(state_loss) + tf.reduce_mean(reward_loss) # Combine mean losses

        grads = tape.gradient(total_loss, self.world_model.trainable_variables)
        # Optional: Clip gradients
        # grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.model_optimizer.apply_gradients(zip(grads, self.world_model.trainable_variables))
        return tf.reduce_mean(state_loss), tf.reduce_mean(reward_loss)


    def learn(self):
        """ Performs learning: updates Q-net and World Model from real data, then plans. """
        if len(self.memory) < self.batch_size:
            return None # Not enough real experience yet

        # --- 1. Learning from Real Experience ---
        mini_batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        mini_batch = [self.memory[i] for i in mini_batch_indices]
        states_np = np.array([t[0] for t in mini_batch], dtype=np.float32)
        actions_np = np.array([t[1] for t in mini_batch], dtype=np.int32)
        rewards_np = np.array([t[2] for t in mini_batch], dtype=np.float32)
        next_states_np = np.array([t[3] for t in mini_batch], dtype=np.float32)
        dones_np = np.array([t[4] for t in mini_batch])

        states_tf = tf.convert_to_tensor(states_np); actions_tf = tf.convert_to_tensor(actions_np)
        rewards_tf = tf.convert_to_tensor(rewards_np); next_states_tf = tf.convert_to_tensor(next_states_np)
        dones_tf = tf.convert_to_tensor(dones_np, dtype=tf.float32) # Ensure float

        # Train Q-network and World Model on the real batch
        q_loss_real_tf = self._train_q_step(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf)
        model_s_loss_tf, model_r_loss_tf = self._train_model_step(states_tf, actions_tf, rewards_tf, next_states_tf)

        # --- 2. Planning (Learning from Simulated Experience) ---
        plan_q_losses = []
        if self.planning_steps > 0 and len(self.memory) > 0: # Need some history to sample from
            for _ in range(self.planning_steps):
                # Sample previously observed states randomly from memory
                sim_indices = np.random.choice(len(self.memory), self.batch_size, replace=True)
                sim_states_np = np.array([self.memory[i][0] for i in sim_indices], dtype=np.float32)
                # Choose random actions for these states
                sim_actions_np = np.random.randint(0, self.action_size, size=self.batch_size, dtype=np.int32)

                sim_states_tf = tf.convert_to_tensor(sim_states_np)
                sim_actions_tf = tf.convert_to_tensor(sim_actions_np)

                # Use world model to predict next state and reward
                # TODO: Pass action if model requires it
                sim_next_states_tf, sim_rewards_tf = self.world_model(sim_states_tf, training=False)
                sim_rewards_tf = tf.squeeze(sim_rewards_tf)
                sim_dones_tf = tf.zeros_like(sim_rewards_tf, dtype=tf.float32) # Assume simulation doesn't end episode

                # Perform Q-update using the simulated transition
                q_loss_plan = self._train_q_step(sim_states_tf, sim_actions_tf, sim_rewards_tf, sim_next_states_tf, sim_dones_tf)
                plan_q_losses.append(q_loss_plan.numpy())

        # --- Decay Epsilon ---
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

        # --- Return Metrics ---
        metrics = {
            'q_loss_real': q_loss_real_tf.numpy(),
            'model_state_loss': model_s_loss_tf.numpy(),
            'model_reward_loss': model_r_loss_tf.numpy(),
            'q_loss_planning_avg': np.mean(plan_q_losses) if plan_q_losses else 0.0, # Use 0 if no planning
            'epsilon': self.epsilon
        }
        return metrics

    # --- Target Model Update ---
    def update_target_model(self):
        """ Copies weights from the online Q-model to the target Q-model. """
        # Use self._log from BaseAgent
        self._log(logging.INFO, f"DynaQAgent updating DQN target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    # --- Save/Load ---
    def save(self, name_base):
        """ Saves DQN models and the World Model. """
        dqn_model_file = name_base + "_dqn.keras"; world_model_file = name_base + "_world.keras"
        self._save_keras_model(self.model, dqn_model_file) # Helper logs
        self._save_keras_model(self.world_model, world_model_file) # Helper logs

    def load(self, name_base):
        """ Loads DQN models and the World Model. """
        dqn_model_file = name_base + "_dqn.keras"; world_model_file = name_base + "_world.keras"
        loaded_dqn = self._load_keras_model(dqn_model_file) # Helper logs
        if loaded_dqn: self.model = loaded_dqn; self.update_target_model(); self._log(logging.INFO, "DynaQ loaded DQN model.")
        loaded_world = self._load_keras_model(world_model_file) # Helper logs
        if loaded_world: self.world_model = loaded_world; self._log(logging.INFO, "DynaQ loaded World model.")

# --- END OF FILE project/agents/dyna_q_agent.py ---
