# --- START OF FILE project/agents/ddpg_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import random
from collections import deque
import logging

from agents.base_agent import BaseAgent
# Adjust path if needed, assuming networks is a sibling of agents
try:
    from networks.tf_networks import build_mlp_actor, build_mlp_critic
except ImportError:
    from ..networks.tf_networks import build_mlp_actor, build_mlp_critic


class DDPGAgent(BaseAgent):
    """ Deep Deterministic Policy Gradient (DDPG) Agent (Adapted for Discrete Actions) """
    def __init__(self, game,
                 lr_actor=0.0001, lr_critic=0.001,
                 gamma=0.99, tau=0.005, # Soft update factor
                 memory_size=100000, batch_size=64,
                 hidden_units=(256, 256), activation='relu',
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995, # For exploration
                 model_path=None, logger=None):

        super().__init__(game, model_path, logger=logger)

        # DDPG assumes vector input
        if not hasattr(game, 'get_state_size'): raise ValueError("DDPGAgent requires game.get_state_size()")
        self.state_size = game.get_state_size()
        self.action_size = game.get_action_space()
        self.input_shape = (self.state_size,)
        self.input_dtype = tf.float32 # Assume vector state is float

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Exploration strategy for discrete actions
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Build Networks
        self.actor = build_mlp_actor(self.input_shape, self.action_size, hidden_units, activation, "DDPG_Actor")
        self.critic = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "DDPG_Critic")
        self.target_actor = build_mlp_actor(self.input_shape, self.action_size, hidden_units, activation, "DDPG_TargetActor")
        self.target_critic = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "DDPG_TargetCritic")

        # Optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = optimizers.Adam(learning_rate=lr_critic)
        self.critic_loss_fn = losses.MeanSquaredError() # Use instantiated loss

        # Replay Buffer
        self.replay_buffer = deque(maxlen=self.memory_size)

        # Initial Target Sync
        self.update_target_model(hard=True)

        if self.model_path_base:
            self.load(self.model_path_base)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state.astype(np.float32), action, np.float32(reward),
                                   next_state.astype(np.float32), done))

    def get_action(self, state, use_exploration=True):
        """ Get action from actor network, add exploration noise if training """
        if use_exploration and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Epsilon-greedy exploration

        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=self.input_dtype)
        action_logits = self.actor(state_tensor, training=False)
        action = tf.argmax(action_logits[0]).numpy() # Deterministic action from actor
        return action

    # @tf.function # <--- TEMPORARILY COMMENT OUT THIS DECORATOR
    def _train_step(self, states, actions, rewards, next_states, dones):
        """ Performs one DDPG training step. """
        # Add print statements for debugging shapes and types if needed
        # tf.print("States shape:", tf.shape(states), "dtype:", states.dtype)
        # tf.print("Actions shape:", tf.shape(actions), "dtype:", actions.dtype)
        # tf.print("Rewards shape:", tf.shape(rewards), "dtype:", rewards.dtype)
        # tf.print("Next states shape:", tf.shape(next_states), "dtype:", next_states.dtype)
        # tf.print("Dones shape:", tf.shape(dones), "dtype:", dones.dtype)

        # Target Q Calculation
        with tf.GradientTape() as critic_tape:
            target_actions_logits = self.target_actor(next_states, training=False)
            target_next_actions = tf.argmax(target_actions_logits, axis=1, output_type=tf.int32) # (b,)

            target_q_all = self.target_critic(next_states, training=False) # (b, num_actions)
            indices = tf.stack([tf.range(tf.shape(target_next_actions)[0], dtype=tf.int32), target_next_actions], axis=1)
            target_next_q = tf.gather_nd(target_q_all, indices) # (b,)

            # Bellman target
            target_q = rewards + self.gamma * target_next_q * (1.0 - dones)

            # Critic Loss
            current_q_all = self.critic(states, training=True) # (b, num_actions)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices) # (b,)

            critic_loss = self.critic_loss_fn(tf.stop_gradient(target_q), current_q) # MSE loss
            critic_loss_mean = tf.reduce_mean(critic_loss)
            # tf.print("Critic Loss:", critic_loss_mean) # Debug print

        # Update Critic
        critic_grads = critic_tape.gradient(critic_loss_mean, self.critic.trainable_variables)
        # Check for None gradients (can happen if loss doesn't depend on variables)
        # if any(g is None for g in critic_grads):
        #    tf.print("Warning: Found None gradient in critic update.")
        # Apply gradients if they exist
        valid_critic_grads = [(g, v) for g, v in zip(critic_grads, self.critic.trainable_variables) if g is not None]
        if valid_critic_grads:
            self.critic_optimizer.apply_gradients(valid_critic_grads)

        # Actor Loss
        with tf.GradientTape() as actor_tape:
            actor_actions_logits = self.actor(states, training=True)
            actor_actions = tf.argmax(actor_actions_logits, axis=1, output_type=tf.int32)

            critic_q_for_actor_all = self.critic(states, training=False) # No gradient through critic here
            actor_indices = tf.stack([tf.range(tf.shape(actor_actions)[0], dtype=tf.int32), actor_actions], axis=1)
            critic_q_for_actor = tf.gather_nd(critic_q_for_actor_all, actor_indices)

            actor_loss = -tf.reduce_mean(critic_q_for_actor)
            # tf.print("Actor Loss:", actor_loss) # Debug print

        # Update Actor
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        # if any(g is None for g in actor_grads):
        #    tf.print("Warning: Found None gradient in actor update.")
        valid_actor_grads = [(g, v) for g, v in zip(actor_grads, self.actor.trainable_variables) if g is not None]
        if valid_actor_grads:
            self.actor_optimizer.apply_gradients(valid_actor_grads)

        # tf.print("Returning from _train_step:", critic_loss_mean, actor_loss)
        return critic_loss_mean, actor_loss


    def learn(self):
        """ Samples batch, performs training step, returns metrics. """
        if len(self.replay_buffer) < self.batch_size:
            return None

        mini_batch = random.sample(self.replay_buffer, self.batch_size)

        states_np = np.array([t[0] for t in mini_batch])
        actions_np = np.array([t[1] for t in mini_batch], dtype=np.int32)
        rewards_np = np.array([t[2] for t in mini_batch], dtype=np.float32)
        next_states_np = np.array([t[3] for t in mini_batch])
        dones_np = np.array([t[4] for t in mini_batch], dtype=np.float32)

        states = tf.convert_to_tensor(states_np, dtype=self.input_dtype)
        actions = tf.convert_to_tensor(actions_np)
        rewards = tf.convert_to_tensor(rewards_np)
        next_states = tf.convert_to_tensor(next_states_np, dtype=self.input_dtype)
        dones = tf.convert_to_tensor(dones_np)

        # Call the potentially non-decorated function
        train_step_output = self._train_step(states, actions, rewards, next_states, dones)

        # Check if the output is None BEFORE unpacking
        if train_step_output is None:
            self._log(logging.ERROR, "_train_step returned None even without @tf.function!")
            # Handle this case, maybe return None or default metrics
            return None
        else:
            critic_loss, actor_loss = train_step_output

        # Soft update target networks
        self.update_target_model()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        metrics = {
            'critic_loss': critic_loss.numpy(),
            'actor_loss': actor_loss.numpy(),
            'epsilon': self.epsilon
        }
        return metrics

    def update_target_model(self, hard=False):
        """ Soft update target networks """
        if hard:
            self._log(logging.DEBUG, f"{type(self).__name__}: Hard updating target models.")
            if self.actor and self.target_actor: self.target_actor.set_weights(self.actor.get_weights())
            if self.critic and self.target_critic: self.target_critic.set_weights(self.critic.get_weights())
        else:
            # Soft update actor
            if self.actor and self.target_actor:
                current_weights = self.actor.get_weights()
                target_weights = self.target_actor.get_weights()
                # Ensure weights are compatible
                if len(current_weights) == len(target_weights):
                    new_weights = [self.tau * cw + (1.0 - self.tau) * tw for cw, tw in zip(current_weights, target_weights)]
                    self.target_actor.set_weights(new_weights)
                else: self._log(logging.ERROR, "Actor weight mismatch during soft update.")
            # Soft update critic
            if self.critic and self.target_critic:
                current_weights = self.critic.get_weights()
                target_weights = self.target_critic.get_weights()
                if len(current_weights) == len(target_weights):
                    new_weights = [self.tau * cw + (1.0 - self.tau) * tw for cw, tw in zip(current_weights, target_weights)]
                    self.target_critic.set_weights(new_weights)
                else: self._log(logging.ERROR, "Critic weight mismatch during soft update.")


    def save(self, name_base):
        if self.actor: self._save_keras_model(self.actor, name_base + "_actor.keras")
        if self.critic: self._save_keras_model(self.critic, name_base + "_critic.keras")

    def load(self, name_base):
        loaded_actor = self._load_keras_model(name_base + "_actor.keras")
        if loaded_actor: self.actor = loaded_actor
        loaded_critic = self._load_keras_model(name_base + "_critic.keras")
        if loaded_critic: self.critic = loaded_critic
        # Sync targets after loading main models
        self.update_target_model(hard=True)

# --- END OF FILE project/agents/ddpg_agent.py ---
