# --- START OF FILE project/agents/cnn_dqn_agent_pytorch.py ---

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
import logging # Import logging

# Import network builder, PER buffer, and base agent
from networks.pytorch_networks import build_cnn_dqn_pytorch # Builder function
from replay_memory import PER_ReplayBuffer
from agents.base_agent import BaseAgent

class CNN_DQN_Agent_PyTorch(BaseAgent):
    """ CNN-DQN Agent (PyTorch) - Uses network builder """
    def __init__(self, game,
                 lr=0.0001, gamma=0.99, batch_size=32,
                 memory_size=100000, target_update_freq=10000, # In frames
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=1000000,
                 noisy_std_init=0.5, # Passed to network builder
                 model_path=None,
                 input_type='screen', # Added for consistency, though PyTorch agent assumes screen
                 logger=None): # <<< ADD logger=None HERE
        # Pass logger to super() call
        super().__init__(game, model_path, logger=logger) # <<< PASS logger HERE

        # Verify game compatibility (assuming screen input for this agent)
        if not hasattr(game,'get_screen_size') or not callable(game.get_screen_size):
            raise ValueError("Game needs 'get_screen_size'.")
        if not hasattr(game,'get_action_space') or not callable(game.get_action_space):
            raise ValueError("Game needs 'get_action_space'.")
        self.input_shape = game.get_screen_size() # (H, W, C) - PyTorch net expects this order
        self.action_size = game.get_action_space()
        self.input_dtype_np = np.uint8 # Assuming screen input is uint8

        self.gamma = gamma; self.batch_size = batch_size; self.target_update_freq = target_update_freq
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log(logging.INFO, f"{type(self).__name__} using device: {self.device}")

        # PER Parameters
        self.alpha = per_alpha; self.beta_start = per_beta_start; self.beta = per_beta_start
        self.beta_frames = per_beta_frames
        self.replay_buffer = PER_ReplayBuffer(memory_size, alpha=self.alpha)
        # frame_idx inherited

        # Build Networks
        # Assuming build_cnn_dqn_pytorch handles the input shape correctly
        self.model = build_cnn_dqn_pytorch(
            self.input_shape, self.action_size, noisy_std_init).to(self.device)
        self.target_model = build_cnn_dqn_pytorch(
            self.input_shape, self.action_size, noisy_std_init).to(self.device)

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none') # Huber loss, element-wise

        self.update_target_model() # Initial sync
        self.target_model.eval() # Set target to eval mode

        # Epsilon not needed (assuming Noisy Nets)
        self.epsilon = 0.0

        if self.model_path_base:
            self.load(self.model_path_base) # Uses helper which logs

    def remember(self, state, action, reward, next_state, done):
        """Stores uint8 screen state in the PER buffer."""
         # Ensure state is uint8 if coming directly from Pygame screen capture
        self.replay_buffer.add((state.astype(self.input_dtype_np),
                                action,
                                np.float32(reward),
                                next_state.astype(self.input_dtype_np),
                                done))

    def get_action(self, state):
        """Selects action using the noisy network."""
        # state is numpy (H, W, C), uint8
        # Convert to tensor, permute to (C, H, W), add batch dim, send to device
        state_tensor = torch.tensor(state, dtype=torch.uint8, device=self.device).permute(2, 0, 1).unsqueeze(0)
        # Set model to train mode for noise generation if using NoisyLinear
        self.model.train()
        with torch.no_grad():
            q_values = self.model(state_tensor)
        # No need to set back to eval unless other layers require it (like BatchNorm)
        # self.model.eval() # Not strictly necessary if only NoisyLinear needs train()
        return q_values.argmax().item() # Get action index

    def update_beta(self):
         """Anneals beta linearly."""
         fraction = min(float(self.frame_idx) / self.beta_frames, 1.0)
         self.beta = self.beta_start + fraction * (1.0 - self.beta_start)

    def learn(self):
        """Samples batch, performs training step, updates priorities, returns metrics."""
        if len(self.replay_buffer) < self.batch_size: return None
        self.update_beta()

        batch, idxs, weights_np = self.replay_buffer.sample(self.batch_size, self.beta)
        if batch is None:
            self._log(logging.WARNING, "PyTorch CNN sampling failed.")
            return None

        # Prepare Tensors
        weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        # Ensure correct numpy dtype before converting to tensor
        states_np = np.array([b[0] for b in batch], dtype=self.input_dtype_np)
        actions_np = np.array([b[1] for b in batch])
        rewards_np = np.array([b[2] for b in batch])
        next_states_np = np.array([b[3] for b in batch], dtype=self.input_dtype_np)
        dones_np = np.array([b[4] for b in batch])

        # Convert numpy arrays to PyTorch tensors
        # Permute image dimensions from (B, H, W, C) to (B, C, H, W)
        states = torch.tensor(states_np, dtype=torch.uint8, device=self.device).permute(0, 3, 1, 2)
        next_states = torch.tensor(next_states_np, dtype=torch.uint8, device=self.device).permute(0, 3, 1, 2)
        actions = torch.tensor(actions_np, dtype=torch.long, device=self.device).unsqueeze(1) # (B, 1) long for gather
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=self.device).unsqueeze(1) # (B, 1) float
        dones = torch.tensor(dones_np, dtype=torch.float32, device=self.device).unsqueeze(1) # (B, 1) float for calculation

        # Double DQN Target Calculation
        with torch.no_grad():
            # Use online model in train mode for action selection (samples noise)
            self.model.train()
            online_next_q = self.model(next_states)
            next_actions = online_next_q.argmax(1).unsqueeze(1) # (B, 1) Indices of best actions

            # Use target model in eval mode for stability
            self.target_model.eval()
            target_next_q = self.target_model(next_states)
            # Gather Q-values from target net corresponding to actions selected by online net
            ddqn_next_val = target_next_q.gather(1, next_actions)
            # Calculate TD target
            target = rewards + self.gamma * ddqn_next_val * (1.0 - dones) # (B, 1)

        # Q-value Prediction and Loss Calculation
        # Set model to train mode for gradient calculation and noise sampling
        self.model.train()
        # Get Q-values from online model for the actions actually taken
        current_q = self.model(states).gather(1, actions) # (B, 1)
        # Calculate element-wise Huber loss
        element_loss = self.loss_fn(current_q, target) # (B, 1)
        # Apply PER Importance Sampling weights and compute mean loss
        loss = (weights * element_loss).mean()

        # Backpropagation
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0) # Clip gradients
        self.optimizer.step() # Update model parameters

        # Update Priorities in PER buffer
        # Calculate TD errors (absolute difference)
        td_errors = (target - current_q).abs().squeeze(1).detach().cpu().numpy() # (B,) detach, move to cpu, convert to numpy
        for i, idx in enumerate(idxs):
            self.replay_buffer.update(idx, td_errors[i])

        # Return metrics
        metrics = {
            'loss': loss.item(), # Get scalar value from tensor
            'mean_td_error': np.mean(td_errors)
            # Beta is updated internally, could return self.beta if needed
        }
        return metrics


    def update_target_model(self):
        """Copies weights from online model to target model."""
        self._log(logging.INFO, f"CNN_DQN_Agent_PyTorch updating target model at frame {self.frame_idx}")
        if self.model and self.target_model:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval() # Ensure target stays in eval mode

    def load(self, name_base):
        """Loads model weights using helper."""
        model_file = name_base + ".pt"
        # Use helper; it handles logging and device mapping
        if self._load_pytorch_model(self.model, model_file, self.device):
             self.update_target_model() # Sync target model after loading online model

    def save(self, name_base):
        """Saves model weights using helper."""
        model_file = name_base + ".pt"
        self._save_pytorch_model(self.model, model_file) # Use helper

# --- END OF FILE project/agents/cnn_dqn_agent_pytorch.py ---
