# --- START OF FILE project/agents/sac_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import tensorflow_probability as tfp
import numpy as np
from collections import deque
import logging
import random  # <--- ADD THIS IMPORT
import pickle  # <--- ADD THIS IMPORT (needed for save/load)
import os      # <--- ADD THIS IMPORT (needed for save/load)

from agents.base_agent import BaseAgent
# Adjust path if needed
try:
    from networks.tf_networks import build_mlp_actor, build_mlp_critic
except ImportError:
    from ..networks.tf_networks import build_mlp_actor, build_mlp_critic

class SACAgent(BaseAgent):
    """ Soft Actor-Critic (SAC) Agent (Adapted for Discrete Actions) """
    def __init__(self, game,
                 lr_actor=0.0003, lr_critic=0.0003, lr_alpha=0.0003, # LRs
                 gamma=0.99, tau=0.005, # Discount and soft update factor
                 alpha=0.2, # Initial temperature OR use None for learned alpha
                 target_entropy=None, # Target entropy for learned alpha
                 memory_size=100000, batch_size=64,
                 hidden_units=(256, 256), activation='relu',
                 model_path=None, logger=None):

        super().__init__(game, model_path, logger=logger)

        # SAC assumes vector input
        if not hasattr(game, 'get_state_size'): raise ValueError("SACAgent requires game.get_state_size()")
        self.state_size = game.get_state_size()
        self.action_size = game.get_action_space()
        self.input_shape = (self.state_size,)
        self.input_dtype = tf.float32

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory_size = memory_size

        # Temperature Alpha (Entropy coefficient)
        self.learn_alpha = alpha is None # Learn alpha if initial value is not provided
        if self.learn_alpha:
            # Logarithm of alpha for numerical stability
            self.log_alpha = tf.Variable(0.0, dtype=tf.float32, name="log_alpha")
            self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp) # Get alpha value by exp(log_alpha)
            self.alpha_optimizer = optimizers.Adam(learning_rate=lr_alpha)
            # Target entropy heuristic: -log(1/|A|) * scale (e.g., scale=0.98)
            if target_entropy is None:
                 # Heuristic target entropy for discrete actions
                 self.target_entropy = -np.log(1.0 / self.action_size) * 0.98
            else:
                 self.target_entropy = float(target_entropy)
            self._log(logging.INFO, f"SAC using learned alpha, target entropy: {self.target_entropy:.4f}")
        else:
            self.alpha = tf.constant(alpha, dtype=tf.float32)
            self.log_alpha = tf.math.log(self.alpha) # Store log for consistency if needed
            self._log(logging.INFO, f"SAC using fixed alpha: {alpha:.4f}")

        # Build Networks (Actor, Critic1, Critic2, Target Critics)
        self.actor = build_mlp_actor(self.input_shape, self.action_size, hidden_units, activation, "SAC_Actor")
        self.critic1 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "SAC_Critic1")
        self.critic2 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "SAC_Critic2")
        self.target_critic1 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "SAC_TargetCritic1")
        self.target_critic2 = build_mlp_critic(self.input_shape, self.action_size, hidden_units, activation, "SAC_TargetCritic2")

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

    def _get_policy_distribution(self, state_tensor):
        """ Get action distribution and log probs from actor """
        # Ensure training=True to compute gradients correctly later if needed inside train_step
        action_logits = self.actor(state_tensor, training=True)
        # Use Categorical for discrete actions
        action_dist = tfp.distributions.Categorical(logits=action_logits)
        return action_dist

    def get_action(self, state, use_exploration=True):
        """ Get action by sampling from the stochastic policy """
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=self.input_dtype)
        # Need distribution without gradients for sampling action only
        action_logits = self.actor(state_tensor, training=False)
        action_dist = tfp.distributions.Categorical(logits=action_logits)

        if use_exploration:
            action = action_dist.sample() # Sample from the distribution
        else:
            # For deterministic evaluation, take the mode (most likely action)
            action = action_dist.mode()

        return action.numpy()[0] # Return action index

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones):
        """ Performs one SAC training step. """
        with tf.GradientTape(persistent=True) as tape:
            # Add log_alpha to watched variables if it's learnable
            if self.learn_alpha:
                 tape.watch(self.log_alpha)

            # --- Critic Loss ---
            # Get next actions and log probs from current policy for next states
            next_action_dist = self._get_policy_distribution(next_states)
            next_actions_sample = next_action_dist.sample() # Sample next actions a'
            next_log_probs = next_action_dist.log_prob(next_actions_sample) # Log prob log pi(a'|s')

            # Target Q values using minimum of target critics
            target_q1_all = self.target_critic1(next_states, training=False)
            target_q2_all = self.target_critic2(next_states, training=False)
            # Indices for gathering Q-values of sampled next_actions
            next_action_indices = tf.stack([tf.range(tf.shape(next_actions_sample)[0], dtype=tf.int32), next_actions_sample], axis=1)
            target_q1 = tf.gather_nd(target_q1_all, next_action_indices)
            target_q2 = tf.gather_nd(target_q2_all, next_action_indices)
            target_q_min = tf.minimum(target_q1, target_q2)

            # Current alpha value (potentially tensor if learned)
            alpha_val = tf.exp(self.log_alpha) if self.learn_alpha else self.alpha

            # Include entropy term in target: Q_target - alpha * log pi(a'|s')
            target_value = target_q_min - alpha_val * next_log_probs
            # Bellman target: r + gamma * target_value * (1 - done)
            target_q = rewards + self.gamma * target_value * (1.0 - dones)
            target_q_detached = tf.stop_gradient(target_q)

            # Current Q estimates from critics for the *actual* actions taken in buffer
            current_q1_all = self.critic1(states, training=True)
            current_q2_all = self.critic2(states, training=True)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            current_q1 = tf.gather_nd(current_q1_all, action_indices)
            current_q2 = tf.gather_nd(current_q2_all, action_indices)

            # Critic losses (MSE)
            critic1_loss = tf.reduce_mean(self.critic_loss_fn(target_q_detached, current_q1))
            critic2_loss = tf.reduce_mean(self.critic_loss_fn(target_q_detached, current_q2))
            critic_total_loss = critic1_loss + critic2_loss # Combined loss for optimization? Often optimized separately. Let's stick to separate.

            # --- Actor Loss ---
            # Get actions and log_probs for the *current* states from the current policy
            current_action_dist = self._get_policy_distribution(states)
            current_actions_sample = current_action_dist.sample() # Sample actions a ~ pi(.|s)
            current_log_probs = current_action_dist.log_prob(current_actions_sample) # Log prob log pi(a|s)

            # Q-values for these *newly sampled* actions from critic 1 (minimum critic often not used here)
            current_q1_for_actor_all = self.critic1(states, training=False) # No gradient through critic needed for actor loss calculation
            # Reuse critic2 as well? Let's stick to Q1 for now as is common
            current_actions_sample_indices = tf.stack([tf.range(tf.shape(current_actions_sample)[0], dtype=tf.int32), current_actions_sample], axis=1)
            current_q1_for_actor = tf.gather_nd(current_q1_for_actor_all, current_actions_sample_indices)

            # Actor objective: mean( alpha * log pi(a|s) - Q1(s,a) )
            actor_loss = tf.reduce_mean(alpha_val * current_log_probs - current_q1_for_actor)

            # --- Alpha Loss (Optional) ---
            alpha_loss = tf.constant(0.0, dtype=tf.float32) # Default if not learning alpha
            if self.learn_alpha:
                # Use log_probs from the policy action sample taken for the actor loss calculation
                # Objective: mean( -log_alpha * (log pi(a|s) + target_entropy) )
                # Use stop_gradient on log_probs as alpha loss shouldn't affect policy parameters
                alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(current_log_probs + self.target_entropy))

        # --- Apply Gradients ---
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))

        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        if self.learn_alpha:
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha]) # Gradient wrt log_alpha variable
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        del tape # Release persistent tape

        # Get current alpha value to return for logging
        current_alpha_val = tf.exp(self.log_alpha) if self.learn_alpha else self.alpha

        return critic1_loss, critic2_loss, actor_loss, alpha_loss, current_alpha_val

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

        c1_loss, c2_loss, a_loss, alpha_loss, current_alpha = self._train_step(
            states, actions, rewards, next_states, dones
        )

        # Soft update target critics
        self.update_target_model()

        metrics = {
            'critic1_loss': c1_loss.numpy(),
            'critic2_loss': c2_loss.numpy(),
            'actor_loss': a_loss.numpy(),
            'alpha_loss': alpha_loss.numpy(),
            'alpha': current_alpha.numpy() # Log current alpha value
        }
        return metrics

    def update_target_model(self, hard=False):
        """ Soft update target critic networks """
        if hard:
            self._log(logging.DEBUG, f"{type(self).__name__}: Hard updating target models.")
            if self.critic1 and self.target_critic1: self.target_critic1.set_weights(self.critic1.get_weights())
            if self.critic2 and self.target_critic2: self.target_critic2.set_weights(self.critic2.get_weights())
        else:
            # Soft update critic 1
            if self.critic1 and self.target_critic1:
                current_weights = self.critic1.get_weights()
                target_weights = self.target_critic1.get_weights()
                if len(current_weights) == len(target_weights):
                    new_weights = [self.tau * cw + (1.0 - self.tau) * tw for cw, tw in zip(current_weights, target_weights)]
                    self.target_critic1.set_weights(new_weights)
                else: self._log(logging.ERROR,"Critic1 weight mismatch in soft update.")

             # Soft update critic 2
            if self.critic2 and self.target_critic2:
                current_weights = self.critic2.get_weights()
                target_weights = self.target_critic2.get_weights()
                if len(current_weights) == len(target_weights):
                    new_weights = [self.tau * cw + (1.0 - self.tau) * tw for cw, tw in zip(current_weights, target_weights)]
                    self.target_critic2.set_weights(new_weights)
                else: self._log(logging.ERROR,"Critic2 weight mismatch in soft update.")
            # No target actor update needed usually in standard SAC

    def save(self, name_base):
        if self.actor: self._save_keras_model(self.actor, name_base + "_actor.keras")
        if self.critic1: self._save_keras_model(self.critic1, name_base + "_critic1.keras")
        if self.critic2: self._save_keras_model(self.critic2, name_base + "_critic2.keras")
        if self.learn_alpha and hasattr(self, 'log_alpha'):
            # Save log_alpha variable value
            state_file = name_base + "_sac_state.pkl"
            _ensure_dir_exists(state_file) # Use helper function
            try:
                 with open(state_file,'wb') as f: pickle.dump({'log_alpha':self.log_alpha.numpy()}, f)
                 self._log(logging.INFO, f"SAC alpha state saved to {state_file}")
            except Exception as e: self._log(logging.ERROR, f"Failed to save SAC alpha state: {e}")


    def load(self, name_base):
        loaded_actor = self._load_keras_model(name_base + "_actor.keras")
        if loaded_actor: self.actor = loaded_actor
        loaded_critic1 = self._load_keras_model(name_base + "_critic1.keras")
        if loaded_critic1: self.critic1 = loaded_critic1
        loaded_critic2 = self._load_keras_model(name_base + "_critic2.keras")
        if loaded_critic2: self.critic2 = loaded_critic2

        # Load log_alpha if learning it
        if self.learn_alpha and hasattr(self, 'log_alpha'):
             state_file = name_base + "_sac_state.pkl"
             if os.path.exists(state_file):
                 try:
                      with open(state_file, 'rb') as f: sac_state = pickle.load(f)
                      if 'log_alpha' in sac_state:
                           self.log_alpha.assign(sac_state['log_alpha'])
                           self._log(logging.INFO, f"Loaded SAC log_alpha: {self.log_alpha.numpy():.4f}")
                      else: self._log(logging.WARNING, f"SAC state file {state_file} exists but 'log_alpha' key not found.")
                 except Exception as e: self._log(logging.ERROR, f"Failed to load SAC alpha state from {state_file}: {e}")
             else: self._log(logging.INFO, f"SAC state file {state_file} not found. Using initial alpha settings.")

        # Sync targets after loading main models
        self.update_target_model(hard=True)


# --- Helper function needed by SAC save ---
def _ensure_dir_exists(file_path):
    """Creates the directory for a file path if it doesn't exist."""
    # This should ideally live in utils.py, but adding here for completeness
    # if utils.py is not accessible directly or to avoid circular imports.
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            # Use logger if available, otherwise print
            logger = logging.getLogger(__name__)
            if logger.hasHandlers():
                 logger.debug(f"Created directory: {directory}")
            else: print(f"Debug: Created directory: {directory}")
        except OSError as e:
            logger = logging.getLogger(__name__)
            if logger.hasHandlers():
                 logger.error(f"Error creating directory {directory}: {e}", exc_info=True)
            else: print(f"Error creating directory {directory}: {e}")

# --- END OF FILE project/agents/sac_agent.py ---
