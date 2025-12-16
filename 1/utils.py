# --- START OF FILE project/utils.py ---

import pickle
import os
import traceback
import numpy as np # May be needed by pickled buffer objects
import logging # Use logging
from collections import deque # Needed for type checking if loading fails

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Helper Function to Ensure Directory Exists ---
def _ensure_dir_exists(file_path):
    """Creates the directory for a file path if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Error creating directory {directory}: {e}", exc_info=True)
            # Depending on severity, you might want to raise the error
            # raise e

# --- Utility Functions ---

def save_training_state(path_base, episode, agent):
    """
    Saves training progress (episode, frame_idx, beta, epsilon, temp, rnd_stats)
    AND the agent's primary experience buffer (either replay_buffer or memory).
    """
    agent_name = type(agent).__name__
    state_file = path_base + "_state.pkl"
    # Use a generic name for the buffer file
    buffer_file = path_base + "_buffer.pkl"

    # --- Save Core State ---
    _ensure_dir_exists(state_file) # Ensure directory exists for state file
    save_data = {
        # Save state for resuming the *next* episode
        'episode': episode + 1,
        'frame_idx': getattr(agent, 'frame_idx', 0),
        # Try common names for beta, epsilon, temperature
        'beta': getattr(agent, 'beta', getattr(agent, 'per_beta', None)),
        'epsilon': getattr(agent, 'epsilon', None),
        'temperature': getattr(agent, 'temperature', None),
        'rnd_stats': getattr(agent, 'intrinsic_reward_stats', None) # For RND agent
        # Add other agent-specific state vars here if needed
    }
    # Remove None values before saving
    save_data = {k: v for k, v in save_data.items() if v is not None}
    try:
        with open(state_file, 'wb') as f:
            pickle.dump(save_data, f)
        logger.debug(f"{agent_name}: Core state saved to {state_file}")
    except Exception as e:
        logger.error(f"{agent_name}: Error saving core state to {state_file}: {e}", exc_info=True)

    # --- Save Experience Buffer (replay_buffer OR memory) ---
    buffer_to_save = None
    buffer_attr_name = None
    if hasattr(agent, 'replay_buffer'):
        buffer_to_save = agent.replay_buffer
        buffer_attr_name = 'replay_buffer'
    elif hasattr(agent, 'memory'): # Check for 'memory' if 'replay_buffer' not found
        buffer_to_save = agent.memory
        buffer_attr_name = 'memory'
    else:
        logger.warning(f"{agent_name}: Agent has no 'replay_buffer' or 'memory' attribute to save.")

    if buffer_to_save is not None and buffer_attr_name is not None:
        _ensure_dir_exists(buffer_file) # Ensure directory exists for buffer file
        buffer_size = 'N/A'
        try:
            buffer_size = len(buffer_to_save)
        except TypeError:
             logger.warning(f"Could not determine size of buffer attribute '{buffer_attr_name}'")

        logger.info(f"{agent_name}: Attempting to save experience buffer ('{buffer_attr_name}', Size: {buffer_size}) to {buffer_file}...")
        try:
            with open(buffer_file, 'wb') as f:
                # Use highest protocol for efficiency with potentially large data
                pickle.dump(buffer_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"{agent_name}: Experience buffer saved successfully.")
        except (AttributeError, TypeError, pickle.PicklingError, OverflowError) as e:
            # Catch potential errors during pickling large/complex objects
            logger.error(f"{agent_name}: Error saving experience buffer ('{buffer_attr_name}') to {buffer_file}: {e}", exc_info=True)
            logger.warning("  Buffer contents will NOT be persisted for this save.")
            # Optionally remove partially written file if error occurred
            if os.path.exists(buffer_file):
                try: os.remove(buffer_file)
                except OSError: pass
        except Exception as e: # Catch any other unexpected errors
             logger.error(f"{agent_name}: Unexpected error saving experience buffer: {e}", exc_info=True)
             if os.path.exists(buffer_file):
                try: os.remove(buffer_file)
                except OSError: pass


def load_training_state(path_base, agent):
    """
    Loads training progress if state file exists.
    Loads the experience buffer if buffer file exists and agent has corresponding attribute.
    Updates agent attributes in-place. Returns start episode.
    """
    state_file = path_base + "_state.pkl"
    buffer_file = path_base + "_buffer.pkl" # Generic buffer file name
    start_episode = 1 # Default start episode
    agent_name = type(agent).__name__

    # --- Load Core State ---
    state_loaded = False
    if os.path.exists(state_file):
        try:
            with open(state_file, 'rb') as f:
                load_data = pickle.load(f)
            # Get start episode (the one to begin *next*)
            start_episode = load_data.get('episode', 1)
            logger.info(f"\n{agent_name}: State file found ({state_file}). Resuming training from episode {start_episode}.")

            # Load frame index
            loaded_frame_idx = load_data.get('frame_idx', 0)
            if hasattr(agent, 'frame_idx'):
                agent.frame_idx = loaded_frame_idx
            else:
                 # If agent doesn't have frame_idx yet, store it temporarily? Risky.
                 # Better ensure BaseAgent or subclasses always define frame_idx.
                 logger.warning(f"{agent_name} instance missing 'frame_idx', cannot restore from state file.")
                 agent.frame_idx = 0 # Default to 0 if missing
            logger.info(f" {agent_name}: Resuming frame index: {agent.frame_idx}")

            # Load beta (checking common attribute names)
            loaded_beta = load_data.get('beta')
            beta_attr = 'per_beta' if hasattr(agent, 'per_beta') else 'beta'
            if hasattr(agent, beta_attr):
                if loaded_beta is not None:
                    setattr(agent, beta_attr, loaded_beta)
                    logger.info(f" {agent_name}: Resuming beta ({beta_attr}): {getattr(agent, beta_attr):.4f}")
                # else: Agent will use its default starting beta

            # Load epsilon
            loaded_epsilon = load_data.get('epsilon')
            if hasattr(agent, 'epsilon'):
                if loaded_epsilon is not None:
                    agent.epsilon = loaded_epsilon
                    logger.info(f" {agent_name}: Resuming epsilon: {agent.epsilon:.4f}")
                # else: Agent will use its default starting epsilon

            # Load temperature
            loaded_temp = load_data.get('temperature')
            if hasattr(agent, 'temperature'):
                if loaded_temp is not None:
                    agent.temperature = loaded_temp
                    logger.info(f" {agent_name}: Resuming temperature: {agent.temperature:.4f}")
                # else: Agent will use its default starting temperature

            # Load RND stats
            loaded_rnd_stats = load_data.get('rnd_stats')
            if hasattr(agent, 'intrinsic_reward_stats'):
                if loaded_rnd_stats is not None:
                    agent.intrinsic_reward_stats = loaded_rnd_stats
                    # Also potentially restore history deque and recalculate mean/std?
                    # This depends on whether rnd_stats includes the history. Assuming it does for now.
                    if hasattr(agent, 'intrinsic_reward_history') and 'history' in loaded_rnd_stats:
                         agent.intrinsic_reward_history = deque(loaded_rnd_stats['history'], maxlen=agent.reward_history_len)
                         agent._update_reward_stats() # Recalculate mean/std
                    logger.info(f" {agent_name}: Resuming RND stats.")
                # else: Agent uses default RND stats

            state_loaded = True # Mark state as successfully loaded

        except (EOFError, pickle.UnpicklingError, KeyError, TypeError) as e:
            logger.error(f"{agent_name}: Error loading or parsing core state from {state_file}: {e}. State reset.", exc_info=True)
            # Reset relevant attributes if loading failed
            agent.frame_idx = 0
            start_episode = 1
            # Reset other potentially loaded values? Agent __init__ should handle defaults.
        except Exception as e:
             logger.error(f"{agent_name}: Unexpected error loading core state from {state_file}: {e}. State reset.", exc_info=True)
             agent.frame_idx = 0
             start_episode = 1
    else:
        logger.info(f"\n{agent_name}: No state file found ({state_file}). Starting fresh state (episode 1, frame 0).")
        # Ensure frame_idx exists even if starting fresh
        if not hasattr(agent, 'frame_idx'): agent.frame_idx = 0

    # --- Load Experience Buffer ---
    buffer_loaded = False
    buffer_attr_name = None
    target_buffer_attr = None

    # Determine which buffer attribute the agent uses
    if hasattr(agent, 'replay_buffer'):
        buffer_attr_name = 'replay_buffer'
        target_buffer_attr = agent.replay_buffer
    elif hasattr(agent, 'memory'):
        buffer_attr_name = 'memory'
        target_buffer_attr = agent.memory

    if buffer_attr_name and os.path.exists(buffer_file):
        logger.info(f"{agent_name}: Attempting to load experience buffer ('{buffer_attr_name}') from {buffer_file}...")
        try:
            with open(buffer_file, 'rb') as f:
                loaded_buffer = pickle.load(f)

            # Basic check: Is the loaded object compatible? (e.g., check type or key attributes)
            # Check if it's a deque or has capacity (for PER buffer)
            if isinstance(loaded_buffer, type(target_buffer_attr)) or \
               (hasattr(loaded_buffer, 'capacity') and hasattr(target_buffer_attr, 'capacity')):
                 # Assign the loaded buffer to the correct agent attribute
                 setattr(agent, buffer_attr_name, loaded_buffer)
                 buffer_size = len(loaded_buffer) if hasattr(loaded_buffer, '__len__') else 'Unknown'
                 logger.info(f" {agent_name}: Experience buffer ('{buffer_attr_name}') loaded successfully (Size: {buffer_size}).")

                 # Optionally sync alpha/beta for PER buffer if they exist on agent
                 if buffer_attr_name == 'replay_buffer' and hasattr(loaded_buffer, 'alpha') and hasattr(agent, 'alpha'):
                      loaded_buffer.alpha = agent.alpha # Sync loaded buffer's alpha with agent's config
                 # Don't sync beta, as agent anneals it based on frame_idx

                 buffer_loaded = True
            else:
                logger.warning(f" {agent_name}: Loaded buffer file ({buffer_file}) type mismatch. Expected type similar to {type(target_buffer_attr)}, got {type(loaded_buffer)}. Using empty buffer.")
                # Keep the initially created empty buffer

        except (pickle.UnpicklingError, EOFError, TypeError, AttributeError, ImportError) as e:
             logger.error(f"{agent_name}: Error unpickling experience buffer ('{buffer_attr_name}') from {buffer_file}: {e}. Using empty buffer.", exc_info=True)
        except Exception as e: # Catch any other unexpected errors
             logger.error(f"{agent_name}: Unexpected error loading experience buffer: {e}. Using empty buffer.", exc_info=True)

    elif buffer_attr_name: # Agent has a buffer attribute, but file doesn't exist
        logger.info(f" {agent_name}: Buffer file not found ({buffer_file}). Starting with empty '{buffer_attr_name}'.")
    else: # Agent doesn't have 'replay_buffer' or 'memory'
         logger.info(f" {agent_name}: Agent has no known buffer attribute ('replay_buffer' or 'memory'). Cannot load buffer.")

    # Final log message about buffer status
    if not buffer_loaded and state_loaded:
         logger.info(f" {agent_name}: Proceeding with loaded state and empty experience buffer.")
    elif not buffer_loaded and not state_loaded:
         logger.info(f" {agent_name}: Proceeding with fresh state and empty experience buffer.")


    return start_episode # Return the starting episode number

# --- END OF FILE project/utils.py ---
