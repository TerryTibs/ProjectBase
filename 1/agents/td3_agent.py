# --- START OF FILE project/agents/td3_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import random
from collections import deque
import logging

from agents.base_agent import BaseAgent
from networks.tf_networks import build_mlp_actor, build_mlp_critic # Reuse builders

class TD3Agent(BaseAgent):
    """ Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent (Adapted for Discrete Actions) """
    def __init__(self, game,
                 lr_actor=0.0001, lr_critic=0.001,
                 gamma=0.99, tau=0.005, # Soft update factor
                 policy_delay=2, # Update actor/targets every N critic updates
                 memory_size=100000, batch_size=64,
                 hidden_units=(256, 256), activation='relu',
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995, # For exploration
                 model_path=None, logger=None):

        super().__init__(game, model_path, logger=logger)

        # TD3 assumes vector input
        if not hasattr(game, 'get_state_size'): raise ValueError("TD3Agent requires game.get_state_size()")
        self.state_size = game.get_state_size()
        self.action_size = game.get_action_space()
        self.input_shape = (self.state_size,)
        self.input_dtype = tf.float32

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.policy_delay = policy_delay
        self.learn_step_counter = 0

        # Exploration strategy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Build Networks (Actor, Critic1, Critic2, Targets)
        self.actor = build_mlp_actor(self.input_shape, self.action_size, hidden_units, activation, "TD3_Actor")
        self.critic1 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "TD3_Critic1")
        self.critic2 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "TD3_Critic2")
        self.target_actor = build_mlp_actor(self.input_shape, self.action_size, hidden_units, activation, "TD3_TargetActor")
        self.target_critic1 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "TD3_TargetCritic1")
        self.target_critic2 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "TD3_TargetCritic2")

        # Optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=lr_actor)
        self.critic1_optimizer = optimizers.Adam(learning_rate=lr_critic)
        self.critic2_optimizer = optimizers.Adam(learning_rate=lr_critic)
        self.critic_loss_fn = losses.MeanSquaredError()

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
        action = tf.argmax(action_logits[0]).numpy() # Deterministic action
        return action

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """ Performs one TD3 training step (updates critics, optionally actor/targets). """
        # Target Q Calculation (using minimum of target critics)
        target_actions_logits = self.target_actor(next_states, training=False)
        target_next_actions = tf.argmax(target_actions_logits, axis=1, output_type=tf.int32) # (b,)

        # Target policy smoothing noise (skipped for discrete action space adaptation)
        # In continuous: noise = tf.clip_by_value(tf.random.normal(tf.shape(target_next_actions)) * policy_noise, -noise_clip, noise_clip)
        # target_next_actions = tf.clip_by_value(target_next_actions + noise, action_low, action_high)

        target_q1_all = self.target_critic1(next_states, training=False) # (b, num_actions)
        target_q2_all = self.target_critic2(next_states, training=False) # (b, num_actions)

        indices = tf.stack([tf.range(tf.shape(target_next_actions)[0], dtype=tf.int32), target_next_actions], axis=1)
        target_next_q1 = tf.gather_nd(target_q1_all, indices) # (b,)
        target_next_q2 = tf.gather_nd(target_q2_all, indices) # (b,)

        target_next_q = tf.minimum(target_next_q1, target_next_q2) # Clipped Double-Q

        # Bellman target
        target_q = rewards + self.gamma * target_next_q * (1.0 - dones)
        target_q_detached = tf.stop_gradient(target_q)

        # Critic 1 Loss & Update
        with tf.GradientTape() as critic1_tape:
            current_q1_all = self.critic1(states, training=True)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            current_q1 = tf.gather_nd(current_q1_all, action_indices)
            critic1_loss = tf.reduce_mean(self.critic_loss_fn(target_q_detached, current_q1))
        critic1_grads = critic1_tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))

        # Critic 2 Loss & Update
        with tf.GradientTape() as critic2_tape:
            current_q2_all = self.critic2(states, training=True)
            # action_indices reused
            current_q2 = tf.gather_nd(current_q2_all, action_indices)
            critic2_loss = tf.reduce_mean(self.critic_loss_fn(target_q_detached, current_q2))
        critic2_grads = critic2_tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        actor_loss_val = tf.constant(0.0) # Default if actor not updated
        # Delayed Actor & Target Updates
        if self.learn_step_counter % self.policy_delay == 0:
            with tf.GradientTape() as actor_tape:
                actor_actions_logits = self.actor(states, training=True)
                actor_actions = tf.argmax(actor_actions_logits, axis=1, output_type=tf.int32)

                # Use Critic1 for actor loss (as in paper)
                critic1_q_for_actor_all = self.critic1(states, training=False) # No gradient through critic
                actor_indices = tf.stack([tf.range(tf.shape(actor_actions)[0], dtype=tf.int32), actor_actions], axis=1)
                critic1_q_for_actor = tf.gather_nd(critic1_q_for_actor_all, actor_indices)

                actor_loss = -tf.reduce_mean(critic1_q_for_actor)
                actor_loss_val = actor_loss # Store for return

            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            # Soft update target networks
            self.update_target_model()

        return critic1_loss, critic2_loss, actor_loss_val


    def learn(self):
        """ Samples batch, performs training step, returns metrics. """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.learn_step_counter += 1
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

        c1_loss, c2_loss, a_loss = self._train_step(states, actions, rewards, next_states, dones)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        metrics = {
            'critic1_loss': c1_loss.numpy(),
            'critic2_loss': c2_loss.numpy(),
            'actor_loss': a_loss.numpy(),
            'epsilon': self.epsilon
        }
        # Only log non-zero actor loss if it was updated this step
        if self.learn_step_counter % self.policy_delay != 0:
             metrics['actor_loss'] = 0.0 # Indicate no update

        return metrics

    def update_target_model(self, hard=False):
        """ Soft update target networks (called only during delayed steps in learn) """
        if hard:
            self._log(logging.DEBUG, f"{type(self).__name__}: Hard updating target models.")
            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic1.set_weights(self.critic1.get_weights())
            self.target_critic2.set_weights(self.critic2.get_weights())
        else:
            # Soft update actor
            current_weights = self.actor.get_weights()
            target_weights = self.target_actor.get_weights()
            new_weights = [self.tau * cw + (1.0 - self.tau) * tw for cw, tw in zip(current_weights, target_weights)]
            self.target_actor.set_weights(new_weights)
            # Soft update critic 1
            current_weights = self.critic1.get_weights()
            target_weights = self.target_critic1.get_weights()
            new_weights = [self.tau * cw + (1.0 - self.tau) * tw for cw, tw in zip(current_weights, target_weights)]
            self.target_critic1.set_weights(new_weights)
             # Soft update critic 2
            current_weights = self.critic2.get_weights()
            target_weights = self.target_critic2.get_weights()
            new_weights = [self.tau * cw + (1.0 - self.tau) * tw for cw, tw in zip(current_weights, target_weights)]
            self.target_critic2.set_weights(new_weights)

    def save(self, name_base):
        self._save_keras_model(self.actor, name_base + "_actor.keras")
        self._save_keras_model(self.critic1, name_base + "_critic1.keras")
        self._save_keras_model(self.critic2, name_base + "_critic2.keras")

    def load(self, name_base):
        loaded_actor = self._load_keras_model(name_base + "_actor.keras")
        if loaded_actor: self.actor = loaded_actor
        loaded_critic1 = self._load_keras_model(name_base + "_critic1.keras")
        if loaded_critic1: self.critic1 = loaded_critic1
        loaded_critic2 = self._load_keras_model(name_base + "_critic2.keras")
        if loaded_critic2: self.critic2 = loaded_critic2
        # Sync targets after loading main models
        self.update_target_model(hard=True)

# --- END OF FILE project/agents/td3_agent.py ---
