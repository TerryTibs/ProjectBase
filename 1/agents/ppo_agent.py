# --- START OF FILE project/agents/ppo_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import tensorflow_probability as tfp
import numpy as np
import os
from collections import deque
import logging # <<< ADD THIS IMPORT

# Import common components
from networks.tf_networks import build_actor_critic_network
from agents.base_agent import BaseAgent

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent (TF)
    - Simplified version using replay buffer and multiple epochs per batch.
    - Does NOT use GAE for this simple version. Uses simple TD advantage.
    """
    def __init__(self, game,
                 input_type='vector', # 'vector', 'grid', or 'screen'
                 lr=0.0003, # Single learning rate for shared optimizer
                 gamma=0.99,
                 clip_epsilon=0.2, # PPO clipping parameter
                 entropy_coeff=0.01,
                 value_loss_coeff=0.5,
                 epochs_per_batch=10, # Number of optimization epochs per data batch
                 memory_size=2048, # Size of experience buffer (often related to batch size)
                 batch_size=64,    # Minibatch size for optimization epochs
                 # Network parameters
                 shared_units=(128,), actor_units=(64,), critic_units=(64,),
                 activation='relu',
                 model_path=None,
                 logger=None): # Accept logger

        super().__init__(game, model_path, logger=logger) # Pass logger
        self.input_type = input_type
        self._get_input_shape_and_dtype() # Call helper

        self.action_size = game.get_action_space()
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.epochs_per_batch = epochs_per_batch
        self.batch_size = batch_size # Minibatch size
        self.memory_buffer_size = memory_size # Steps to collect

        # Build Actor-Critic Network
        self.ac_model = build_actor_critic_network(
            self.input_shape, self.action_size, shared_units,
            actor_units, critic_units, activation, self.input_type, name="PPO_ActorCritic"
        )

        # Single Optimizer
        self.optimizer = optimizers.Adam(learning_rate=lr)
        # Instantiate loss function(s) if needed (MSE often used for critic)
        self.critic_loss_fn = losses.MeanSquaredError()

        # Experience Buffer (simple list for PPO batch)
        self.memory = []

        # Load model if path exists
        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

    def _get_input_shape_and_dtype(self):
        """ Determine input shape and dtype based on input_type """
        # (Logic remains the same)
        if self.input_type == 'vector':
            if not hasattr(self.game, 'get_state_size') or not callable(self.game.get_state_size): raise ValueError("PPO vector input requires game.get_state_size()")
            self.input_shape = (self.game.get_state_size(),); self.input_dtype = tf.float32
        elif self.input_type == 'grid':
            if not hasattr(self.game, 'get_grid_shape') or not callable(self.game.get_grid_shape): raise ValueError("PPO grid input requires game.get_grid_shape()")
            self.input_shape = self.game.get_grid_shape(); self.input_dtype = tf.float32
        elif self.input_type == 'screen':
            if not hasattr(self.game, 'get_screen_size') or not callable(self.game.get_screen_size): raise ValueError("PPO screen input requires game.get_screen_size()")
            self.input_shape = self.game.get_screen_size(); self.input_dtype = tf.uint8
        else: raise ValueError(f"Invalid input_type '{self.input_type}'")
        # Use self._log which checks if logger exists
        self._log(logging.INFO, f"PPO using {self.input_type} input. Shape: {self.input_shape}, Dtype: {self.input_dtype}")


    def remember(self, state, action, reward, next_state, done):
        """ Stores experience along with needed policy info (log_prob, value). """
        # (Logic remains the same)
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=self.input_dtype)
        action_logits, V_s = self.ac_model(state_tensor, training=False)
        log_action_probs = tf.nn.log_softmax(action_logits)
        action_int = int(action)
        log_prob_action = log_action_probs[0, action_int]
        expected_np_dtype = np.float32 if self.input_dtype == tf.float32 else np.uint8
        self.memory.append((state.astype(expected_np_dtype), action_int, np.float32(reward), next_state.astype(expected_np_dtype), done, log_prob_action.numpy(), tf.squeeze(V_s).numpy()))

    def get_action(self, state):
        """ Selects action stochastically based on the policy. """
        # (Logic remains the same)
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=self.input_dtype)
        action_logits, _ = self.ac_model(state_tensor, training=False)
        action_probs = tf.nn.softmax(action_logits)
        action_distribution = tfp.distributions.Categorical(probs=action_probs)
        action = action_distribution.sample()
        return int(action.numpy()[0])

    def _calculate_advantages_simple(self, rewards, values, next_values, dones):
        """ Calculates simple TD advantages: A = r + gamma*V(s')*(1-d) - V(s) """
        # (Logic remains the same)
        advantages = rewards + self.gamma * next_values * (1.0 - dones) - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        return advantages

    # Training step executed multiple times per batch
    @tf.function
    def _train_step_ppo_epoch(self, states, actions, old_log_probs, advantages, td_targets):
        """ Performs one epoch of PPO optimization on a minibatch. """
        with tf.GradientTape() as tape:
            action_logits, values = self.ac_model(states, training=True)
            values = tf.squeeze(values)
            new_log_probs_all = tf.nn.log_softmax(action_logits)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            new_log_probs = tf.gather_nd(new_log_probs_all, action_indices)
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            surrogate1 = ratio * tf.stop_gradient(advantages)
            surrogate2 = clipped_ratio * tf.stop_gradient(advantages)
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            # <<< Use instantiated critic loss function >>>
            critic_loss = self.critic_loss_fn(tf.stop_gradient(td_targets), values) # Element-wise or reduced based on init
            scaled_critic_loss = self.value_loss_coeff * tf.reduce_mean(critic_loss) # Ensure mean reduction
            action_probs = tf.nn.softmax(action_logits)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            entropy_loss = -self.entropy_coeff * tf.reduce_mean(entropy)
            total_loss = actor_loss + scaled_critic_loss + entropy_loss

        grads = tape.gradient(total_loss, self.ac_model.trainable_variables)
        # Optional: Clip gradients
        # grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.ac_model.trainable_variables))

        # Return mean metrics
        return tf.reduce_mean(actor_loss), tf.reduce_mean(critic_loss), tf.reduce_mean(entropy)


    def learn(self):
        """ Collects a batch, calculates advantages, and performs PPO updates. """
        if len(self.memory) < self.memory_buffer_size:
            return None

        # (Prepare data remains the same)
        batch = self.memory; states_np = np.array([t[0] for t in batch]); actions_np = np.array([t[1] for t in batch], dtype=np.int32)
        rewards_np = np.array([t[2] for t in batch], dtype=np.float32); next_states_np = np.array([t[3] for t in batch])
        dones_np = np.array([t[4] for t in batch], dtype=np.float32); old_log_probs_np = np.array([t[5] for t in batch], dtype=np.float32)
        values_np = np.array([t[6] for t in batch], dtype=np.float32)
        states = tf.convert_to_tensor(states_np, dtype=self.input_dtype); actions = tf.convert_to_tensor(actions_np, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards_np, dtype=tf.float32); next_states = tf.convert_to_tensor(next_states_np, dtype=self.input_dtype)
        dones = tf.convert_to_tensor(dones_np, dtype=tf.float32); old_log_probs = tf.convert_to_tensor(old_log_probs_np, dtype=tf.float32)
        values = tf.convert_to_tensor(values_np, dtype=tf.float32)

        # (Calculate advantages and targets remains the same)
        _, next_values = self.ac_model(next_states, training=False); next_values = tf.squeeze(next_values)
        advantages = self._calculate_advantages_simple(rewards, values, next_values, dones)
        td_targets = rewards + self.gamma * next_values * (1.0 - dones)

        # --- PPO Optimization Epochs ---
        actor_losses, critic_losses, entropies = [], [], []
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_log_probs, advantages, td_targets))
        dataset = dataset.shuffle(buffer_size=self.memory_buffer_size).batch(self.batch_size)

        for _ in range(self.epochs_per_batch):
            for batch_data in dataset:
                s_batch, a_batch, olp_batch, adv_batch, tdt_batch = batch_data
                a_loss, c_loss, ent = self._train_step_ppo_epoch(s_batch, a_batch, olp_batch, adv_batch, tdt_batch)
                actor_losses.append(a_loss.numpy())
                critic_losses.append(c_loss.numpy())
                entropies.append(ent.numpy())

        self.memory.clear() # Clear buffer after training on it

        # Return average metrics
        metrics = {
            'actor_loss': np.mean(actor_losses) if actor_losses else None,
            'critic_loss': np.mean(critic_losses) if critic_losses else None,
            'entropy': np.mean(entropies) if entropies else None,
        }
        return metrics

    def update_target_model(self):
        self._log(logging.DEBUG, "PPOAgent: No target model update needed.")
        pass

    def save(self, name_base):
        model_file = name_base + "_ppo_ac.keras"
        self._save_keras_model(self.ac_model, model_file) # Logs via helper

    def load(self, name_base):
        model_file = name_base + "_ppo_ac.keras"
        custom_objects = {}
        loaded_model = self._load_keras_model(model_file, custom_objects=custom_objects) # Logs via helper
        if loaded_model:
            self.ac_model = loaded_model
            self._log(logging.INFO, "Re-initializing optimizer after loading PPO model weights.")
            # Ideally re-initialize optimizer here if LRs are stored

# --- END OF FILE project/agents/ppo_agent.py ---
