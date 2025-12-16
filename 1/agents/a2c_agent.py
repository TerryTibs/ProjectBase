# --- START OF FILE project/agents/a2c_agent.py ---

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError # Import specific loss class
import tensorflow_probability as tfp
import numpy as np
import os
from collections import deque
import logging # Import logging

# Import common components
from networks.tf_networks import build_actor_critic_network
from agents.base_agent import BaseAgent

class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) Agent (TF).
    - Uses a shared network for actor and critic.
    - Uses a single optimizer for combined loss.
    - Uses a simple deque as an experience buffer for batch updates.
    """
    def __init__(self, game,
                 input_type='vector', # 'vector', 'grid', or 'screen'
                 lr=0.0003,           # Single learning rate for the combined optimizer
                 gamma=0.99,
                 entropy_coeff=0.01,  # Weight for entropy bonus in actor loss
                 value_loss_coeff=0.5,# Weight for critic loss component
                 memory_size=5000,    # Buffer size (can be smaller for A2C)
                 batch_size=64,       # Batch size for learning updates
                 # Network parameters
                 shared_units=(128,), actor_units=(64,), critic_units=(64,),
                 activation='relu',
                 model_path=None,
                 logger=None): # Accept logger

        super().__init__(game, model_path, logger=logger) # Pass logger to base
        self.input_type = input_type
        self._get_input_shape_and_dtype() # Helper sets self.input_shape/dtype

        self.action_size = game.get_action_space()
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.batch_size = batch_size

        # Build the combined Actor-Critic Network
        self.ac_model = build_actor_critic_network(
            self.input_shape, self.action_size, shared_units,
            actor_units, critic_units, activation, self.input_type, name="A2C_ActorCritic"
        )

        # Use a single optimizer for the combined model
        self.optimizer = optimizers.Adam(learning_rate=lr)

        # Instantiate Loss Function for the Critic
        self.critic_loss_fn = MeanSquaredError()

        # Experience Buffer (using deque)
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size # Store max size

        # Load model if path exists
        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

    def _get_input_shape_and_dtype(self):
        """ Determines input shape and dtype based on input_type from the game object. """
        if self.input_type == 'vector':
            if not hasattr(self.game, 'get_state_size') or not callable(self.game.get_state_size): raise ValueError("A2C vector input requires game.get_state_size()")
            self.input_shape = (self.game.get_state_size(),); self.input_dtype = tf.float32
        elif self.input_type == 'grid':
            if not hasattr(self.game, 'get_grid_shape') or not callable(self.game.get_grid_shape): raise ValueError("A2C grid input requires game.get_grid_shape()")
            self.input_shape = self.game.get_grid_shape(); self.input_dtype = tf.float32
        elif self.input_type == 'screen':
            if not hasattr(self.game, 'get_screen_size') or not callable(self.game.get_screen_size): raise ValueError("A2C screen input requires game.get_screen_size()")
            self.input_shape = self.game.get_screen_size(); self.input_dtype = tf.uint8
        else: raise ValueError(f"Invalid input_type '{self.input_type}'")
        self._log(logging.INFO, f"A2C using {self.input_type} input. Shape: {self.input_shape}, Dtype: {self.input_dtype}")

    def remember(self, state, action, reward, next_state, done):
        """ Stores experience, ensuring correct numpy dtype. """
        expected_np_dtype = np.float32 if self.input_dtype == tf.float32 else np.uint8
        self.memory.append((state.astype(expected_np_dtype),
                            action,
                            np.float32(reward),
                            next_state.astype(expected_np_dtype),
                            done))

    def get_action(self, state):
        """ Selects action stochastically based on the policy network. """
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=self.input_dtype)
        # Get action logits from the actor head
        action_logits, _ = self.ac_model(state_tensor, training=False)
        # Convert logits to probabilities - squeeze batch dim before softmax
        action_probs_tensor = tf.nn.softmax(tf.squeeze(action_logits, axis=0))
        action_probs = action_probs_tensor.numpy() # Get numpy array of probs (shape: num_actions,)

        # Log probabilities at DEBUG level
        self._log(logging.DEBUG, f"Action Probs: {[f'{p:.3f}' for p in action_probs]}")

        # Create a categorical distribution from probabilities
        action_distribution = tfp.distributions.Categorical(probs=action_probs_tensor) # Use tensor probs

        # Sample ONE action. This returns a scalar tensor (shape ())
        action_tensor = action_distribution.sample()

        # Convert scalar tensor directly to Python int
        action_int = int(action_tensor.numpy())

        # Log chosen action at DEBUG level
        self._log(logging.DEBUG, f"Chosen Action: {action_int}")

        return action_int

    @tf.function
    def _train_step_tf(self, states, actions, rewards, next_states, dones):
        """
        Performs a single A2C training step with combined loss.
        Calculates losses and applies gradients using the single optimizer.
        Returns individual loss components for monitoring.
        """
        # --- Calculate TD Target and Advantage (No Gradients Needed Here) ---
        _, V_s = self.ac_model(states, training=False)      # V(s)
        _, V_s_prime = self.ac_model(next_states, training=False) # V(s')
        V_s = tf.squeeze(V_s, axis=-1)           # Squeeze last dim -> shape (batch_size,)
        V_s_prime = tf.squeeze(V_s_prime, axis=-1) # Squeeze last dim -> shape (batch_size,)

        # Calculate TD Target: T = r + gamma * V(s') * (1 - done)
        td_target = rewards + self.gamma * V_s_prime * (1.0 - dones) # shape (batch_size,)

        # Calculate Advantage: A = T - V(s)
        delta = td_target - V_s # shape (batch_size,)

        # --- Calculate Combined Loss and Gradients in ONE tape ---
        with tf.GradientTape() as tape:
            # Recompute model outputs needed for losses *within* the tape for gradient tracking
            action_logits, V_s_for_loss = self.ac_model(states, training=True)
            V_s_for_loss = tf.squeeze(V_s_for_loss, axis=-1) # Squeeze last dim -> shape (batch_size,)

            # --- Critic Loss Component ---
            critic_loss = self.critic_loss_fn(tf.stop_gradient(td_target), V_s_for_loss)
            scaled_critic_loss = self.value_loss_coeff * tf.reduce_mean(critic_loss)

            # --- Actor Loss Component ---
            log_probs = tf.nn.log_softmax(action_logits)
            action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
            log_probs_taken = tf.gather_nd(log_probs, action_indices)
            actor_loss_term = -log_probs_taken * tf.stop_gradient(delta)

            # --- Entropy Bonus Component ---
            action_probs = tf.nn.softmax(action_logits)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            entropy_loss_term = -entropy

            # --- Combine Actor & Entropy Objectives ---
            actor_objective = tf.reduce_mean(actor_loss_term + self.entropy_coeff * entropy_loss_term)

            # --- Total Combined Loss for the Optimizer ---
            total_loss = actor_objective + scaled_critic_loss

        # --- Apply Gradients ---
        grads = tape.gradient(total_loss, self.ac_model.trainable_variables)
        # Optional gradient clipping
        # grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, self.ac_model.trainable_variables))

        # --- Return Individual Loss Components (Mean over Batch) for Logging ---
        return tf.reduce_mean(critic_loss), tf.reduce_mean(actor_loss_term), tf.reduce_mean(entropy)

    def learn(self):
        """ Samples a batch from memory, performs training step, returns metrics. """
        if len(self.memory) < self.batch_size:
            return None # Not enough experience

        # Sample a random minibatch of experiences
        mini_batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        mini_batch = [self.memory[i] for i in mini_batch_indices]

        # Unpack batch data into numpy arrays
        states_np = np.array([t[0] for t in mini_batch])
        actions_np = np.array([t[1] for t in mini_batch], dtype=np.int32)
        rewards_np = np.array([t[2] for t in mini_batch], dtype=np.float32)
        next_states_np = np.array([t[3] for t in mini_batch])
        dones_np = np.array([t[4] for t in mini_batch], dtype=np.float32) # Ensure float for TF calcs

        # Convert numpy arrays to TensorFlow tensors with the correct dtype
        states = tf.convert_to_tensor(states_np, dtype=self.input_dtype)
        actions = tf.convert_to_tensor(actions_np, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards_np, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states_np, dtype=self.input_dtype)
        dones = tf.convert_to_tensor(dones_np, dtype=tf.float32)

        # Perform the training step using the collected batch
        mean_critic_loss_tf, mean_actor_loss_term_tf, mean_entropy_tf = self._train_step_tf(
            states, actions, rewards, next_states, dones
        )

        # Return metrics as a dictionary (convert tensors to numpy for logging)
        metrics = {
            'critic_loss': mean_critic_loss_tf.numpy(),
            'actor_loss_term': mean_actor_loss_term_tf.numpy(), # Policy gradient component
            'entropy': mean_entropy_tf.numpy()
        }
        return metrics

    def update_target_model(self):
        """ Standard A2C does not use target networks. """
        self._log(logging.DEBUG, "A2CAgent: update_target_model called, but not used.")
        pass

    def save(self, name_base):
        """ Saves the single ActorCritic model. """
        model_file = name_base + "_ac.keras"
        # Use helper which handles directory creation and logging
        # include_optimizer=True saves the state of the single self.optimizer
        self._save_keras_model(self.ac_model, model_file)

    def load(self, name_base):
        """ Loads the ActorCritic model. """
        model_file = name_base + "_ac.keras"
        custom_objects = {} # Add custom objects if network uses them
        # Load model weights only (compile=False in helper)
        loaded_model = self._load_keras_model(model_file, custom_objects=custom_objects)
        if loaded_model:
            self.ac_model = loaded_model
            # Optimizer state is not loaded by default with compile=False
            self._log(logging.INFO, "A2C model weights loaded. Optimizer state requires re-initialization or separate loading.")

# --- END OF FILE project/agents/a2c_agent.py ---
