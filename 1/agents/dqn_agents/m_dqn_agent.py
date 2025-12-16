# --- START OF FILE project/agents/m_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import random
import logging # Import logging

# Import common components
from networks.tf_networks import build_dense_dqn # Can use simple DQN network
from replay_memory import PER_ReplayBuffer # Can use PER
from agents.base_agent import BaseAgent

class MDQNAgent(BaseAgent):
    """ Munchausen DQN (M-DQN) Agent (TF). """
    def __init__(self, game,
                 # M-DQN specific parameters
                 m_alpha=0.9, # Log-policy term scaling (entropy temperature)
                 m_tau=0.03,  # Target Q-value log-policy term scaling
                 clip_value_min=-1.0, # Clipping for log-policy term
                 clip_value_max=0.0,
                 # Standard DQN parameters
                 lr=0.0001, gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 hidden_units=(64, 64), activation='relu',
                 model_path=None,
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size

        # M-DQN parameters
        self.m_alpha = m_alpha; self.m_tau = m_tau
        self.clip_min = clip_value_min; self.clip_max = clip_value_max

        # Store hyperparameters
        self.gamma = gamma; self.learning_rate = lr; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.per_alpha = per_alpha; self.per_beta_start = per_beta_start; self.per_beta = per_beta_start
        self.per_beta_frames = per_beta_frames

        # Initialize PER buffer
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.per_alpha)
        # frame_idx inherited

        # Build simple dense networks
        self.model = build_dense_dqn((self.state_size,), self.action_size, hidden_units, activation, name="MDQN_Online")
        self.target_model = build_dense_dqn((self.state_size,), self.action_size, hidden_units, activation, name="MDQN_Target")

        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate, clipnorm=10.0)
        self.loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE) # Element-wise loss
        self.update_target_model() # Initial sync

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

        # Epsilon greedy needed for exploration
        self.epsilon_start = 1.0; self.epsilon_end = 0.01; self.epsilon_decay = 0.9999
        self.epsilon = self.epsilon_start

    def remember(self, state, action, reward, next_state, done):
        """Stores experience."""
        self.replay_buffer.add((state.astype(np.float32),
                                action,
                                np.float32(reward),
                                next_state.astype(np.float32),
                                done))

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        # Exploit
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        q_values = self.model(state_input, training=False).numpy() # Use __call__
        return np.argmax(q_values[0])

    def update_beta(self): # PER Beta annealing
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.per_beta_frames, 1.0)
        self.per_beta = self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs M-DQN training step."""
        # --- Calculate Log-Policy Terms (using current online model Q-values) ---
        # No gradients needed for these Q-values, they form part of the target
        current_q_online_all_no_grad = tf.stop_gradient(self.model(states, training=False))
        # Log-softmax: log( softmax(Q(s,.) / tau) ) = Q(s,.)/tau - logsumexp(Q(s,.)/tau)
        log_softmax_current = current_q_online_all_no_grad / self.m_tau - \
                              tf.reduce_logsumexp(current_q_online_all_no_grad / self.m_tau, axis=1, keepdims=True)

        # --- Calculate Next State Log-Policy & Softmax Policy ---
        next_q_online_all_no_grad = tf.stop_gradient(self.model(next_states, training=False))
        log_softmax_next = next_q_online_all_no_grad / self.m_tau - \
                           tf.reduce_logsumexp(next_q_online_all_no_grad / self.m_tau, axis=1, keepdims=True)
        # Softmax policy pi(a'|s') = softmax(Q_online(s',.) / tau)
        policy_next = tf.nn.softmax(next_q_online_all_no_grad / self.m_tau, axis=1)

        # --- Calculate Munchausen Target ---
        # Get target network Q-values for the next state: Q_target(s', .)
        next_q_target_all = self.target_model(next_states, training=False)
        # Calculate expected target Q-value under the softmax policy: E_{a'~pi}[Q_target(s', a')]
        expected_next_q_target = tf.reduce_sum(policy_next * next_q_target_all, axis=1)
        # Calculate expected log-policy term: E_{a'~pi}[tau * log pi(a'|s')]
        entropy_term_next = self.m_tau * tf.reduce_sum(policy_next * log_softmax_next, axis=1)
        # Combine for the next state value term used in the target
        next_value_term = expected_next_q_target + entropy_term_next # Note: Paper derivation detail

        # --- Calculate Modified Reward ---
        # Get log-policy of the action actually taken: log pi(a|s)
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        action_indices_gather = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
        log_policy_action_taken = tf.gather_nd(log_softmax_current, action_indices_gather)
        # Clip the log-policy term
        clipped_log_policy = tf.clip_by_value(log_policy_action_taken, self.clip_min, self.clip_max)
        # Modified Munchausen reward: R' = R + alpha * clip(log pi(a|s))
        modified_reward = tf.cast(rewards, tf.float32) + self.m_alpha * clipped_log_policy

        # --- Final Target Calculation ---
        # Target = R' + gamma * (E[Q_target] + E[tau*log pi]) * (1 - done)
        target = modified_reward + self.gamma * next_value_term * (1.0 - tf.cast(dones, tf.float32))

        # --- Loss Calculation and Gradient Update ---
        with tf.GradientTape() as tape:
            # Online model prediction *with* gradients enabled for the actions taken
            current_q_all_online = self.model(states, training=True)
            current_q_action_taken = tf.gather_nd(current_q_all_online, action_indices_gather)
            # Calculate element-wise Huber loss against the Munchausen target
            element_loss = self.loss_fn(tf.stop_gradient(target), current_q_action_taken)
            # Apply PER Importance Sampling weights
            weighted_element_loss = tf.cast(weights, tf.float32) * element_loss
            # Reduce to scalar loss for the optimizer
            loss = tf.reduce_mean(weighted_element_loss)

        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Calculate TD errors for PER update (using the Munchausen target)
        td_errors = tf.abs(target - current_q_action_taken)
        # Return TD errors and loss
        return td_errors, loss

    def learn(self):
        """Samples batch, performs M-DQN training, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.per_beta)
        if batch is None:
            self._log(logging.WARNING, "M-DQN sampling failed.")
            return None

        # Prepare batch data
        states_np=np.array([b[0] for b in batch],dtype=np.float32); actions_np=np.array([b[1] for b in batch],dtype=np.int32)
        rewards_np=np.array([b[2] for b in batch],dtype=np.float32); next_states_np=np.array([b[3] for b in batch],dtype=np.float32)
        dones_np=np.array([b[4] for b in batch],dtype=np.float32) # Float for TF

        # Convert to tensors
        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np); weights_tf=tf.convert_to_tensor(weights_np,dtype=tf.float32)

        # Execute train step
        td_errors_tf, loss_tf = self._train_step_tf(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf)

        # Update priorities
        td_errors_np = td_errors_tf.numpy()
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idxs[i], td_errors_np[i])

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # Return metrics
        metrics = {
            'loss': loss_tf.numpy(),
            'mean_td_error': np.mean(td_errors_np),
            'beta': self.per_beta, # Use self.per_beta
            'epsilon': self.epsilon
        }
        return metrics

    def update_target_model(self):
        """Copies weights to target model."""
        self._log(logging.INFO, f"MDQNAgent updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".keras"
        # Standard Dense layers, no custom objects needed by default
        loaded_model = self._load_keras_model(model_file) # Uses helper which logs
        if loaded_model:
            self.model = loaded_model
            self.update_target_model()

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".keras"
        self._save_keras_model(self.model, model_file) # Uses helper which logs

# --- END OF FILE project/agents/m_dqn_agent.py ---
