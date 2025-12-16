# --- START OF FILE project/agents/dqn_agents/icm_dqn_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import os
import random
import logging
from collections import deque

# Import common components
from networks.tf_networks import (build_dense_dqn, build_icm_encoder, # Needs creating
                                 build_icm_forward_model, build_icm_inverse_model)
from replay_memory import PER_ReplayBuffer # Can use PER or simple deque
from agents.base_agent import BaseAgent

class ICMDQNAgent(BaseAgent):
    """ DQN Agent with Intrinsic Curiosity Module (ICM) (TF). """
    def __init__(self, game,
                 # ICM specific parameters
                 feature_dim=128, # Output dimension of ICM encoder phi(s)
                 icm_hidden_units=(64, 64), # Hidden units for ICM networks
                 intrinsic_reward_scale=0.01, # eta in paper
                 policy_loss_weight=1.0, # beta in paper (weight for inverse model loss)
                 forward_loss_weight=0.2, # lambda in paper (weight for forward model loss vs policy loss)
                 # Standard DQN parameters
                 lr_dqn=0.0001, lr_icm=0.0001, # Separate LRs
                 gamma=0.99, batch_size=64,
                 memory_size=100000, target_update_freq=5000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 dqn_hidden_units=(64, 64), activation='relu',
                 model_path=None,
                 logger=None):
        super().__init__(game, model_path, logger=logger)

        if not hasattr(game, 'state_size') or not hasattr(game, 'action_size'):
             raise ValueError("Game must have 'state_size' and 'action_size'.")
        self.state_size = game.state_size; self.action_size = game.action_size
        self.input_shape = (self.state_size,) # Assuming vector state for ICM nets

        # ICM parameters
        self.feature_dim = feature_dim
        self.intrinsic_scale = intrinsic_reward_scale
        self.policy_loss_weight = policy_loss_weight
        self.forward_loss_weight = forward_loss_weight

        # Store hyperparameters
        self.gamma = gamma; self.lr_dqn=lr_dqn; self.lr_icm=lr_icm; self.batch_size = batch_size
        self.memory_size = memory_size; self.target_update_freq = target_update_freq
        self.per_alpha = per_alpha; self.per_beta_start = per_beta_start; self.per_beta = per_beta_start
        self.per_beta_frames = per_beta_frames

        # Using PER buffer - stores (s, a, r_extrinsic, s', d)
        self.replay_buffer = PER_ReplayBuffer(self.memory_size, alpha=self.per_alpha)
        # frame_idx inherited

        # Build DQN Networks
        self.model = build_dense_dqn(self.input_shape, self.action_size, dqn_hidden_units, activation, name="ICM_DQN_Online")
        self.target_model = build_dense_dqn(self.input_shape, self.action_size, dqn_hidden_units, activation, name="ICM_DQN_Target")

        # Build ICM Networks
        self.icm_encoder = build_icm_encoder(self.input_shape, self.feature_dim, icm_hidden_units, activation, name="ICM_Encoder")
        self.icm_forward_model = build_icm_forward_model(self.feature_dim, self.action_size, icm_hidden_units, activation, name="ICM_Forward")
        self.icm_inverse_model = build_icm_inverse_model(self.feature_dim, self.action_size, icm_hidden_units, activation, name="ICM_Inverse")

        # Optimizers and Losses
        self.dqn_optimizer = optimizers.Adam(learning_rate=self.lr_dqn, clipnorm=10.0)
        self.dqn_loss_fn = losses.Huber(reduction=tf.keras.losses.Reduction.NONE) # Element-wise Huber
        # Combined optimizer for ICM components (encoder, forward, inverse)
        self.icm_optimizer = optimizers.Adam(learning_rate=self.lr_icm, clipnorm=10.0)
        self.forward_loss_fn = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) # Element-wise MSE for features
        self.inverse_loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) # Action prediction

        self.update_target_model() # Sync DQN target model

        if self.model_path_base:
             self.load(self.model_path_base) # Uses helper which logs

        # Epsilon greedy needed for DQN part
        self.epsilon_start = 1.0; self.epsilon_end = 0.01; self.epsilon_decay = 0.9999
        self.epsilon = self.epsilon_start

    def remember(self, state, action, reward, next_state, done):
        """Stores only extrinsic reward. Intrinsic reward calculated during learning."""
        self.replay_buffer.add((state.astype(np.float32), action, np.float32(reward),
                                next_state.astype(np.float32), done))

    def get_action(self, state):
        """Epsilon-greedy action selection using the DQN model."""
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_size)
        state_input = np.expand_dims(np.asarray(state, dtype=np.float32), axis=0)
        q_values = self.model(state_input, training=False).numpy()
        return np.argmax(q_values[0])

    def update_beta(self):
        """Anneals PER beta."""
        fraction = min(float(self.frame_idx) / self.per_beta_frames, 1.0)
        self.per_beta = self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones, weights):
        """Performs ICM-DQN training step, updating DQN and ICM networks."""
        # --- ICM Loss Calculation & Intrinsic Reward ---
        with tf.GradientTape() as icm_tape:
            # Encode states and next_states into feature space (phi)
            features = self.icm_encoder(states, training=True) # Track grads for encoder
            next_features = self.icm_encoder(next_states, training=True) # Track grads for encoder

            # Predict next features using Forward Model
            # Need actions one-hot encoded for the forward model input
            actions_one_hot = tf.one_hot(tf.cast(actions, tf.int32), depth=self.action_size, dtype=tf.float32)
            predicted_next_features = self.icm_forward_model([features, actions_one_hot], training=True)

            # Calculate Forward Loss (Prediction Error) -> Raw Intrinsic Reward
            forward_loss_elements = self.forward_loss_fn(tf.stop_gradient(next_features), predicted_next_features) # Shape (b,)
            intrinsic_rewards_raw = 0.5 * forward_loss_elements # eta/2 * ||phi(s') - f(phi(s),a)||^2
            forward_loss = tf.reduce_mean(forward_loss_elements)

            # Predict actions using Inverse Model
            predicted_action_logits = self.icm_inverse_model([features, next_features], training=True)

            # Calculate Inverse Loss (Action Prediction)
            inverse_loss_elements = self.inverse_loss_fn(tf.cast(actions, tf.int32), predicted_action_logits) # Shape (b,)
            inverse_loss = tf.reduce_mean(inverse_loss_elements)

            # Combined ICM Loss (weighted sum)
            # Loss = (1-beta)*L_I + beta*L_F
            # L_I = inverse_loss, L_F = forward_loss
            # Policy loss weight = beta -> self.policy_loss_weight
            # Forward loss weight = 1 - beta -> not directly used, scale factor lambda applied later
            icm_loss = (1.0 - self.policy_loss_weight) * inverse_loss + self.policy_loss_weight * forward_loss
            # Paper uses L = L_I + L_F and then optimizer update combines policy loss with (lambda * ICM_Loss)
            # Let's follow paper: Total loss for ICM optimizer is L_I + L_F
            # Total loss for policy optimizer is L_Policy - lambda * L_Curiosity(L_F)

            # Here, we train ICM nets based on combined loss L_I + L_F
            total_icm_loss = inverse_loss + forward_loss

        # --- Apply gradients to ICM networks (Encoder, Forward, Inverse) ---
        icm_trainable_vars = self.icm_encoder.trainable_variables + \
                             self.icm_forward_model.trainable_variables + \
                             self.icm_inverse_model.trainable_variables
        icm_grads = icm_tape.gradient(total_icm_loss, icm_trainable_vars)
        self.icm_optimizer.apply_gradients(zip(icm_grads, icm_trainable_vars))

        # --- DQN Training using Combined Reward ---
        # Combine extrinsic reward with scaled intrinsic reward
        # Use stop_gradient as intrinsic reward influences target but shouldn't flow back to ICM nets via DQN loss
        combined_rewards = tf.cast(rewards, tf.float32) + self.intrinsic_scale * tf.stop_gradient(intrinsic_rewards_raw)

        # Standard DDQN target calculation using combined reward
        online_next_q = self.model(next_states, training=False)
        next_actions_dqn = tf.argmax(online_next_q, axis=1, output_type=tf.int32)
        target_next_q = self.target_model(next_states, training=False)
        batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, next_actions_dqn], axis=1)
        ddqn_next_val = tf.gather_nd(target_next_q, gather_indices)
        target = combined_rewards + self.gamma * ddqn_next_val * (1.0 - tf.cast(dones, tf.float32))

        # DQN Loss Calculation and Gradient Update
        with tf.GradientTape() as dqn_tape:
            current_q_all = self.model(states, training=True)
            action_indices_dqn = tf.stack([batch_indices, tf.cast(actions, tf.int32)], axis=1)
            current_q = tf.gather_nd(current_q_all, action_indices_dqn)
            element_loss = self.dqn_loss_fn(tf.stop_gradient(target), current_q)
            weighted_element_loss = tf.cast(weights, tf.float32) * element_loss
            dqn_loss = tf.reduce_mean(weighted_element_loss)

        # Apply gradients to DQN online model
        dqn_grads = dqn_tape.gradient(dqn_loss, self.model.trainable_variables)
        self.dqn_optimizer.apply_gradients(zip(dqn_grads, self.model.trainable_variables))

        # TD Errors for PER update (based on combined reward target)
        td_errors = tf.abs(target - current_q)

        # Return TD errors, DQN loss, Forward loss (proxy for avg intrinsic reward), Inverse loss
        return td_errors, dqn_loss, forward_loss, inverse_loss # Return individual ICM losses

    def learn(self):
        """Samples batch, performs ICM+DQN training, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()
        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.per_beta)
        if batch is None: self._log(logging.WARNING,"ICM-DQN sampling failed."); return None

        states_np=np.array([b[0] for b in batch],dtype=np.float32); actions_np=np.array([b[1] for b in batch],dtype=np.int32)
        rewards_np=np.array([b[2] for b in batch],dtype=np.float32); next_states_np=np.array([b[3] for b in batch],dtype=np.float32)
        dones_np=np.array([b[4] for b in batch],dtype=np.float32) # Float for TF

        states_tf=tf.convert_to_tensor(states_np); actions_tf=tf.convert_to_tensor(actions_np)
        rewards_tf=tf.convert_to_tensor(rewards_np); next_states_tf=tf.convert_to_tensor(next_states_np)
        dones_tf=tf.convert_to_tensor(dones_np); weights_tf=tf.convert_to_tensor(weights_np,dtype=tf.float32)

        td_errors_tf, dqn_loss_tf, forward_loss_tf, inverse_loss_tf = self._train_step_tf(
            states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf, weights_tf
        )

        td_errors_np = td_errors_tf.numpy()
        for i, idx in enumerate(idxs): self.replay_buffer.update(idxs[i], td_errors_np[i])

        if self.epsilon > self.epsilon_end: self.epsilon *= self.epsilon_decay

        metrics = {
            'dqn_loss': dqn_loss_tf.numpy(),
            'icm_forward_loss': forward_loss_tf.numpy(), # Avg intrinsic reward proxy
            'icm_inverse_loss': inverse_loss_tf.numpy(),
            'mean_td_error': np.mean(td_errors_np),
            'beta': self.per_beta, 'epsilon': self.epsilon,
        }
        return metrics

    def update_target_model(self):
        """Copies weights for DQN target model."""
        self._log(logging.INFO, f"ICMDQNAgent updating DQN target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
        # No target network for ICM components

    def load(self, name_base):
        """Loads DQN and ICM models."""
        dqn_model_file = name_base + "_dqn.keras"
        enc_file = name_base + "_icm_encoder.keras"
        fwd_file = name_base + "_icm_forward.keras"
        inv_file = name_base + "_icm_inverse.keras"
        # Load DQN
        loaded_dqn = self._load_keras_model(dqn_model_file);
        if loaded_dqn: self.model = loaded_dqn; self.update_target_model()
        # Load ICM components
        loaded_enc = self._load_keras_model(enc_file);
        if loaded_enc: self.icm_encoder = loaded_enc
        loaded_fwd = self._load_keras_model(fwd_file);
        if loaded_fwd: self.icm_forward_model = loaded_fwd
        loaded_inv = self._load_keras_model(inv_file);
        if loaded_inv: self.icm_inverse_model = loaded_inv

    def save(self, name_base):
        """Saves DQN and ICM models."""
        dqn_model_file = name_base + "_dqn.keras"
        enc_file = name_base + "_icm_encoder.keras"
        fwd_file = name_base + "_icm_forward.keras"
        inv_file = name_base + "_icm_inverse.keras"
        # Save DQN
        self._save_keras_model(self.model, dqn_model_file)
        # Save ICM components
        self._save_keras_model(self.icm_encoder, enc_file)
        self._save_keras_model(self.icm_forward_model, fwd_file)
        self._save_keras_model(self.icm_inverse_model, inv_file)

# --- END OF FILE project/agents/dqn_agents/icm_dqn_agent.py ---