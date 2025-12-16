# --- START OF FILE project/agents/base_agent.py ---

from abc import ABC, abstractmethod
import os
import tensorflow as tf
import torch
import traceback
import logging # Import logging
import numpy as np # Often needed

class BaseAgent(ABC):
    """ Abstract base class for common agent functionalities. """
    # <<< FIX: Add logger=None to the __init__ signature >>>
    def __init__(self, game, model_path_base, logger=None):
        self.game = game
        # model_path_base NOW includes the sub-directory, e.g., "models/snake_dqn/snake_dqn_model"
        self.model_path_base = model_path_base
        self.model = None
        self.target_model = None
        self.frame_idx = 0 # Common counter
        # <<< FIX: Assign the passed logger or create a default one >>>
        self.logger = logger if logger else logging.getLogger(type(self).__name__) # Use specific logger or default based on class name

    # Helper method for logging (checks if logger exists)
    def _log(self, level, msg, *args, **kwargs):
        """Logs a message using the agent's logger."""
        # No change needed here, uses self.logger which is now correctly assigned
        if self.logger:
            # Use logger's built-in log method which handles level constants
            self.logger.log(level, msg, *args, **kwargs)

    @abstractmethod
    def get_action(self, state):
        """Selects an action based on the current state."""
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        """Stores experience."""
        pass

    @abstractmethod
    def learn(self):
        """Performs a learning step. Should return metrics dict or None."""
        pass

    @abstractmethod
    def update_target_model(self):
        """Updates the target network."""
        # Log message example (can be called by subclasses via super() or they implement their own)
        self._log(logging.INFO, f"BaseAgent: Requesting target model update at frame {self.frame_idx}")
        pass # Subclasses implement actual weight copy

    @abstractmethod
    def save(self, name_base):
        """Saves the agent's model(s)."""
        pass

    @abstractmethod
    def load(self, name_base):
        """Loads the agent's model(s)."""
        pass

    # --- Directory and Save/Load Helpers ---
    # (No changes needed in _ensure_dir_exists, _save_keras_model, _load_keras_model, etc.)
    # They now use self._log() which correctly references the assigned logger.

    def _ensure_dir_exists(self, file_path):
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                self._log(logging.DEBUG, f"Created directory: {directory}") # Use helper
            except OSError as e:
                 self._log(logging.ERROR, f"Error creating directory {directory}: {e}", exc_info=True) # Use helper

    def _save_keras_model(self, model, file_path):
        self._ensure_dir_exists(file_path)
        self._log(logging.INFO, f"Saving Keras model to {file_path}") # Use helper
        try:
            model.save(file_path, include_optimizer=True)
            self._log(logging.INFO, f"Keras model saved successfully.") # Use helper
        except Exception as e:
             self._log(logging.ERROR, f"Error saving Keras model to {file_path}: {e}", exc_info=True) # Use helper

    def _load_keras_model(self, file_path, custom_objects=None):
        self._log(logging.INFO, f"Attempting to load Keras model from {file_path}") # Use helper
        if os.path.exists(file_path):
            try:
                loaded_model = tf.keras.models.load_model(file_path, custom_objects=custom_objects, compile=False)
                self._log(logging.INFO, f"Keras model loaded successfully from {file_path}.") # Use helper
                return loaded_model
            except Exception as e:
                self._log(logging.ERROR, f"Error loading Keras model from {file_path}: {e}", exc_info=True) # Use helper
                self._log(logging.WARNING,"Proceeding with potentially uninitialized model.") # Use helper
                return None
        else:
            self._log(logging.INFO, f"Keras model file not found at {file_path}. Model not initialized from file.") # Use helper
            return None

    def _save_pytorch_model(self, model, file_path):
        self._ensure_dir_exists(file_path)
        self._log(logging.INFO, f"Saving PyTorch model state_dict to {file_path}") # Use helper
        try:
            torch.save(model.state_dict(), file_path)
            self._log(logging.INFO, f"PyTorch model saved successfully.") # Use helper
        except Exception as e:
            self._log(logging.ERROR, f"Error saving PyTorch model to {file_path}: {e}", exc_info=True) # Use helper

    def _load_pytorch_model(self, model, file_path, device):
        self._log(logging.INFO, f"Attempting to load PyTorch model state_dict from {file_path}") # Use helper
        if os.path.exists(file_path):
            try:
                model.load_state_dict(torch.load(file_path, map_location=device))
                model.to(device)
                self._log(logging.INFO, f"PyTorch model loaded successfully from {file_path}.") # Use helper
                return True
            except Exception as e:
                self._log(logging.ERROR, f"Error loading PyTorch model from {file_path}: {e}", exc_info=True) # Use helper
                self._log(logging.WARNING,"Proceeding with potentially uninitialized model.") # Use helper
                return False
        else:
            self._log(logging.INFO, f"PyTorch model file not found at {file_path}. Model not initialized from file.") # Use helper
            return False
# --- END OF FILE project/agents/base_agent.py ---
