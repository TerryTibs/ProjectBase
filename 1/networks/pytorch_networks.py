import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------- NoisyLinear Class (PyTorch) - CORRECTED ---------------- #
# Using Factorized Gaussian Noise version
class NoisyLinear(nn.Module):
    """
    Noisy linear module for Factorized Gaussian Noise exploration (PyTorch).
    Corrected noise generation shape.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters (means) - Shape: (out_features, in_features)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # Learnable parameters (standard deviations) - Shape: (out_features, in_features)
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features)) # Shape: (out_features,)

        # *** CORRECTED BUFFERS ***
        # Non-learnable buffers for noise factors
        # Noise for input features (matches in_features dimension)
        self.register_buffer('noise_in', torch.empty(1, in_features))
        # Noise for output features (matches out_features dimension)
        self.register_buffer('noise_out', torch.empty(out_features, 1))
        # Bias noise derived from output noise
        self.register_buffer('bias_noise', torch.empty(out_features))
        # *** END CORRECTION ***

        self.reset_parameters()
        self.reset_noise() # Initialize noise buffers

    def reset_parameters(self):
        """Initialize the learnable parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        # Initialize sigma parameters
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features)) # Bias std uses out_features

    def _scale_noise(self, size):
        """Generate noise using f(x) = sign(x) * sqrt(|x|)."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    # *** CORRECTED NOISE GENERATION ***
    def reset_noise(self):
        """Generate new noise samples using the factorized scheme."""
        # Generate noise based on input dimension, shape (1, in_features)
        epsilon_in = self._scale_noise((1, self.in_features))
        # Generate noise based on output dimension, shape (out_features, 1)
        epsilon_out = self._scale_noise((self.out_features, 1))

        # Update noise buffers
        self.noise_in.copy_(epsilon_in)
        self.noise_out.copy_(epsilon_out)
        # Bias noise uses the scaled output noise directly (squeezed)
        self.bias_noise.copy_(epsilon_out.squeeze(1)) # Squeeze dim 1 -> shape (out_features,)
    # *** END CORRECTION ***


    def forward(self, x):
        """Forward pass with noisy weights and biases."""
        if self.training:
            self.reset_noise()

        # *** CORRECTED WEIGHT EPSILON CALCULATION ***
        # Combine noise factors using outer product: (out, 1) * (1, in) -> (out, in)
        weight_epsilon = self.noise_out * self.noise_in
        # *** END CORRECTION ***

        # Calculate noisy weights and biases
        # weight_sigma shape: (out, in) ; weight_epsilon shape: (out, in) -> Compatible
        noisy_weight = self.weight_mu + self.weight_sigma * weight_epsilon
        # bias_sigma shape: (out,) ; bias_noise shape: (out,) -> Compatible
        noisy_bias = self.bias_mu + self.bias_sigma * self.bias_noise

        # Apply linear transformation
        return F.linear(x, noisy_weight, noisy_bias)


# ---------------- CNN_DQN Network Class (PyTorch) ---------------- #
class PyTorch_CNN_DQN(nn.Module):
    """CNN-based DQN model architecture using PyTorch."""
    def __init__(self, input_shape, num_actions, noisy_std_init=0.5):
        super(PyTorch_CNN_DQN, self).__init__()
        h, w, c = input_shape; self.input_shape_torch = (c, h, w); self.num_actions = num_actions
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_shape_torch[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        # Calculate flattened size
        self.fc_input_size = self._get_conv_out(self.input_shape_torch)
        # Fully connected layers using (corrected) NoisyLinear
        self.fc_layers = nn.Sequential(
            NoisyLinear(self.fc_input_size, 512, std_init=noisy_std_init), nn.ReLU(),
            NoisyLinear(512, num_actions, std_init=noisy_std_init))
        print(f"\n--- Built Network: PyTorch_CNN_DQN ---")
        print(f"  Input Shape (C, H, W): {self.input_shape_torch}")
        print(f"  Conv Out Size: {self.fc_input_size}")
        print(f"  Output Actions: {num_actions}"); print("-" * 40)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape); conv_out = self.conv_layers(dummy_input)
            return int(np.prod(conv_out.shape[1:]))

    def forward(self, x):
        x = x.float() / 255.0; conv_out = self.conv_layers(x)
        flattened = torch.flatten(conv_out, start_dim=1); q_values = self.fc_layers(flattened)
        return q_values

# --- Builder Function ---
def build_cnn_dqn_pytorch(input_shape, num_actions, noisy_std_init=0.5):
    """Builds and returns an instance of the PyTorch CNN DQN."""
    return PyTorch_CNN_DQN(input_shape, num_actions, noisy_std_init)
