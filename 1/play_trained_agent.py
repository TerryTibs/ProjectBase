# --- START OF FILE project/play_trained_agent.py ---

import pygame
import os
import time
import traceback
import numpy as np
import logging
import argparse

# --- Game Import ---
from snake_game import SnakeGame

# --- Agent Imports ---
# (Imports remain the same as the previous version - include all agents)
from agents.dqn_agents.dqn_agent import DQNAgent
from agents.dqn_agents.dqn_per_agent import DQN_PER_Agent
from agents.dqn_agents.dqn_rainbow_agent import DQNRainbowAgent
from agents.dqn_agents.dqn_rainbow_lstm_agent import DQNRainbowLSTMAgent
from agents.dqn_agents.dqn_lstm_boltze_agent import DQNLSTMBoltzeAgent
from agents.dqn_agents.cnn_dqn_agent_pytorch import CNN_DQN_Agent_PyTorch
from agents.dqn_agents.n_step_rainbow_agent import NStepRainbowAgent
from agents.dqn_agents.c51_dqn_agent import C51DQNAgent
from agents.dqn_agents.m_dqn_agent import MDQNAgent
from agents.dqn_agents.bootstrapped_dqn_agent import BootstrappedDQNAgent
from agents.dqn_agents.rnd_dqn_agent import RNDDQNAgent
from agents.dqn_agents.dyna_q_agent import DynaQAgent
from agents.dqn_agents.qr_dqn_agent import QRDQNAgent
from agents.dqn_agents.icm_dqn_agent import ICMDQNAgent
from agents.dqn_agents.dueling_soft_dqn_agent import DuelingSoftDQNAgent
from agents.dqn_agents.averaged_dqn_agent import AveragedDQNAgent
from agents.dqn_agents.noisy_dqn_agent import NoisyDQNAgent
from agents.dqn_agents.n_step_c51_dqn_agent import NStepC51DQNAgent
from agents.a2c_agent import A2CAgent
from agents.ppo_agent import PPOAgent
from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent

# Conditional TF CNN agent import
try:
    from agents.dqn_agents.cnn_dqn_agent_tf import CNN_DQN_Agent_TF
    CNN_DQN_Agent_TF_Available = True
except ImportError:
    CNN_DQN_Agent_TF = None
    CNN_DQN_Agent_TF_Available = False


# --- Basic Logging Configuration ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# --- Game and Agent Mapping (Keep these up-to-date) ---
GAMES = {1: "snake"}
GAME_OBJECTS = {"snake": SnakeGame}

AGENT_KEYS = { # Number -> Key String mapping
    # Original DQN Agents (1-13, excluding non-DQNs 14, 15, 17, 18)
    1: "dqn", 2: "dqn_per", 3: "dqn_rainbow", 4: "dqn_rainbow_lstm",
    5: "dqn_lstm_boltze", 6: "cnn_dqn_tf_screen", 7: "cnn_dqn_pytorch",
    8: "n_step_rainbow", 9: "c51_dqn", 10: "m_dqn", 11: "bootstrapped_dqn",
    12: "rnd_dqn", 13: "cnn_dqn_tf_grid", 16: "dyna_q_vector",
    # Policy Gradient / Others (Remain outside DQN folder)
    14: "a2c_vector", 15: "ppo_vector", 17: "a2c_cnn_grid", 18: "ppo_cnn_screen",
    # New DQN Agents (Starting from 19)
    19: "qr_dqn",
    20: "icm_dqn",
    21: "dueling_soft_dqn",
    22: "averaged_dqn",
    23: "noisy_dqn",
    24: "n_step_c51_dqn",
    # New Actor-Critic Style Agents (Starting from 25)
    25: "ddpg",
    26: "td3",
    27: "sac",
}

AGENT_CLASSES = { # Key String -> Class Object mapping
    # Original DQN Agents
    "dqn": DQNAgent, "dqn_per": DQN_PER_Agent, "dqn_rainbow": DQNRainbowAgent,
    "dqn_rainbow_lstm": DQNRainbowLSTMAgent, "dqn_lstm_boltze": DQNLSTMBoltzeAgent,
    "cnn_dqn_pytorch": CNN_DQN_Agent_PyTorch,
    "cnn_dqn_tf_screen": CNN_DQN_Agent_TF if CNN_DQN_Agent_TF_Available else None,
    "cnn_dqn_tf_grid": CNN_DQN_Agent_TF if CNN_DQN_Agent_TF_Available else None,
    "n_step_rainbow": NStepRainbowAgent, "c51_dqn": C51DQNAgent, "m_dqn": MDQNAgent,
    "bootstrapped_dqn": BootstrappedDQNAgent, "rnd_dqn": RNDDQNAgent, "dyna_q_vector": DynaQAgent,
    # Policy Gradient / Others
    "a2c_vector": A2CAgent, "ppo_vector": PPOAgent, "a2c_cnn_grid": A2CAgent, "ppo_cnn_screen": PPOAgent,
    # New DQN Agents
    "qr_dqn": QRDQNAgent,
    "icm_dqn": ICMDQNAgent,
    "dueling_soft_dqn": DuelingSoftDQNAgent,
    "averaged_dqn": AveragedDQNAgent,
    "noisy_dqn": NoisyDQNAgent,
    "n_step_c51_dqn": NStepC51DQNAgent,
    # New Actor-Critic Style Agents
    "ddpg": DDPGAgent,
    "td3": TD3Agent,
    "sac": SACAgent,
}


# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Run a trained RL agent for Snake.")
# Make agent_key optional now
parser.add_argument("-a", "--agent_key", type=str, default=None, help="Agent key (e.g., 'dqn', 'ppo_vector', 'sac') to load (prompts if not specified).")
parser.add_argument("-g", "--game_key", type=str, default="snake", help="Game key (default: 'snake')")
parser.add_argument("-n", "--num_episodes", type=int, default=10, help="Number of episodes to run (default: 10)")
parser.add_argument("-fps", type=int, default=15, help="Rendering FPS during playback (default: 15)")
parser.add_argument("-l", "--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set logging level (default: INFO)")

args = parser.parse_args()

log_level_numeric = getattr(logging, args.loglevel.upper(), logging.INFO)
logging.getLogger().setLevel(log_level_numeric)
logger.info(f"Logging level set to: {args.loglevel.upper()}")

# --- Main Playback Function (remains the same) ---
def play_agent(game_key, agent_key, num_episodes, target_fps):
    logger.info(f"--- Starting Playback ---")
    logger.info(f" Game: {game_key}")
    logger.info(f" Agent: {agent_key}")
    logger.info(f" Episodes: {num_episodes}")
    logger.info(f" Target FPS: {target_fps}")
    logger.info("-" * 25)

    if game_key not in GAME_OBJECTS:
        logger.critical(f"Invalid game key: {game_key}")
        return
    GameClass = GAME_OBJECTS[game_key]
    game = GameClass(render=True) # Force render=True for playback
    game.fps = target_fps
    logger.info("Game initialized for rendering.")

    if agent_key not in AGENT_CLASSES or AGENT_CLASSES[agent_key] is None:
        logger.critical(f"Agent class for key '{agent_key}' not found or unavailable.")
        if hasattr(game, 'quit'): game.quit()
        return
    AgentClass = AGENT_CLASSES[agent_key]

    # Determine model path based on agent type (DQN vs others)
    is_dqn_agent = agent_key in [
        "dqn", "dqn_per", "dqn_rainbow", "dqn_rainbow_lstm",
        "dqn_lstm_boltze", "cnn_dqn_tf_screen", "cnn_dqn_pytorch",
        "n_step_rainbow", "c51_dqn", "m_dqn", "bootstrapped_dqn",
        "rnd_dqn", "cnn_dqn_tf_grid", "dyna_q_vector",
        "qr_dqn", "icm_dqn", "dueling_soft_dqn", "averaged_dqn",
        "noisy_dqn", "n_step_c51_dqn"
    ]

    models_base_dir = "models"
    agent_name_for_path = agent_key.replace(':','_') # Sanitize key
    if is_dqn_agent:
        agent_folder_name = f"{game_key}_{agent_name_for_path}"
        agent_model_dir = os.path.join(models_base_dir, "dqn_agents", agent_folder_name)
    else: # For A2C, PPO, DDPG, TD3, SAC etc.
        agent_folder_name = f"{game_key}_{agent_name_for_path}"
        agent_model_dir = os.path.join(models_base_dir, agent_folder_name)

    model_path_base = os.path.join(agent_model_dir, f"{game_key}_{agent_name_for_path}_model")
    logger.info(f" Expecting model files at base path: {model_path_base}")

    # Determine expected input type
    agent_state_type = "vector"; agent_init_kwargs = {}
    if agent_key.startswith("cnn_dqn_tf"): agent_init_kwargs['input_type'] = 'grid' if agent_key.endswith("_grid") else 'screen'; agent_state_type = agent_init_kwargs['input_type']
    elif agent_key.startswith("cnn_dqn_pytorch"): agent_init_kwargs['input_type'] = 'screen'; agent_state_type = 'screen'
    elif agent_key.startswith("a2c_cnn") or agent_key.startswith("ppo_cnn"): agent_init_kwargs['input_type'] = 'grid' if agent_key.endswith("_grid") else 'screen'; agent_state_type = agent_init_kwargs['input_type']
    elif agent_key.startswith("a2c") or agent_key.startswith("ppo") or agent_key.startswith("dyna_q"): agent_init_kwargs['input_type'] = 'vector'; agent_state_type = 'vector'
    # Assume DDPG/TD3/SAC use 'vector' unless modified
    elif agent_key in ["ddpg", "td3", "sac"]: agent_state_type = 'vector'
    logger.info(f" Agent requires state type: {agent_state_type.upper()}")

    # Check game compatibility
    if agent_state_type == 'screen':
         if not (hasattr(game, 'get_screen') and hasattr(game, 'get_screen_size')):
             logger.critical("Agent expects screen state, but game doesn't provide get_screen/get_screen_size methods.")
             game.quit(); return
    elif agent_state_type == 'grid':
         if not (hasattr(game, 'get_grid_state') and hasattr(game, 'get_grid_shape')):
             logger.critical("Agent expects grid state, but game doesn't provide get_grid_state/get_grid_shape methods.")
             game.quit(); return
    else: # vector
         if not (hasattr(game, 'get_state') and hasattr(game, 'get_state_size')):
             logger.critical("Agent expects vector state, but game doesn't provide get_state/get_state_size methods.")
             game.quit(); return

    agent = None # Initialize agent to None
    try:
        # Initialize agent, passing logger
        agent = AgentClass(game=game, model_path=model_path_base, logger=logger, **agent_init_kwargs)
        logger.info("Loading trained model weights...")
        agent.load(model_path_base) # Agent handles internal logging

        # --- Set Inference Mode ---
        # Disable epsilon-greedy if applicable
        if hasattr(agent, 'epsilon'):
             agent.epsilon = 0.0; logger.info(" Set agent epsilon to 0.0.")

        # Set PyTorch models to eval mode
        if hasattr(agent, 'model') and hasattr(agent.model, 'eval'): # PyTorch DQN/A2C/PPO etc.
             agent.model.eval(); logger.info(" Set PyTorch model to eval mode.")
        if hasattr(agent, 'ac_model') and hasattr(agent.ac_model, 'eval'): # PyTorch ActorCritic
             agent.ac_model.eval(); logger.info(" Set PyTorch AC model to eval mode.")
        if hasattr(agent, 'actor') and hasattr(agent.actor, 'eval'): # PyTorch Actor
            agent.actor.eval(); logger.info(" Set PyTorch Actor model to eval mode.")
        if hasattr(agent, 'critic') and hasattr(agent.critic, 'eval'): # PyTorch Critic
            agent.critic.eval(); logger.info(" Set PyTorch Critic model to eval mode.")
        if hasattr(agent, 'critic1') and hasattr(agent.critic1, 'eval'): # PyTorch Critic1
            agent.critic1.eval(); logger.info(" Set PyTorch Critic1 model to eval mode.")
        if hasattr(agent, 'critic2') and hasattr(agent.critic2, 'eval'): # PyTorch Critic2
            agent.critic2.eval(); logger.info(" Set PyTorch Critic2 model to eval mode.")
        # Set other specific models to eval if needed (e.g., avg_model, ICM, RND)
        if hasattr(agent, 'avg_model') and hasattr(agent.avg_model, 'eval'): agent.avg_model.eval()
        if hasattr(agent, 'icm_encoder') and hasattr(agent.icm_encoder, 'eval'): agent.icm_encoder.eval()
        if hasattr(agent, 'icm_forward_model') and hasattr(agent.icm_forward_model, 'eval'): agent.icm_forward_model.eval()
        if hasattr(agent, 'icm_inverse_model') and hasattr(agent.icm_inverse_model, 'eval'): agent.icm_inverse_model.eval()
        if hasattr(agent, 'rnd_predictor_net') and hasattr(agent.rnd_predictor_net, 'eval'): agent.rnd_predictor_net.eval()
        # Note: TF models generally don't need a separate .eval() call

        logger.info(f"Agent '{agent_key}' initialized and model loaded.")

    except Exception as e:
        logger.critical(f"Error initializing or loading agent '{agent_key}': {e}", exc_info=True)
        game.quit(); return

    # --- Helper to get state ---
    def _get_current_state(game_instance, state_type):
        """Safely gets the state based on the required type."""
        try:
            if state_type == "screen": return game_instance.get_screen()
            elif state_type == "grid": return game_instance.get_grid_state()
            else: return game_instance.get_state()
        except Exception as e:
            logger.error(f"Error getting state type '{state_type}' from game.", exc_info=True)
            # Return a dummy state or re-raise? Returning dummy might hide issues.
            # For playback, it's probably better to crash.
            raise RuntimeError(f"Failed to get state type '{state_type}'") from e


    # --- Playback Loop ---
    total_scores = []
    try:
        for episode in range(1, num_episodes + 1):
            logger.info(f"--- Starting Episode {episode}/{num_episodes} ---")
            game.reset()
            state = _get_current_state(game, agent_state_type)

            done = False
            episode_reward = 0.0
            step = 0

            while not done:
                quit_attempt = False
                # Handle Pygame events (required for window interaction and closing)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                         logger.warning("Quit event received during playback.")
                         quit_attempt = True; break
                if quit_attempt:
                    done = True
                    # Ensure game knows it's over if quit event happens mid-step
                    if hasattr(game, 'game_over'): game.game_over = True
                    break

                # Agent selects action (use exploration=False for inference)
                action = None
                try:
                    if hasattr(agent, 'get_action') and callable(agent.get_action):
                        # Check if get_action accepts 'use_exploration' argument
                        import inspect # Needed to inspect parameters
                        action_params = inspect.signature(agent.get_action).parameters
                        if 'use_exploration' in action_params:
                             action = agent.get_action(state, use_exploration=False)
                        else:
                             action = agent.get_action(state) # Default call
                    else:
                         logger.error("Agent object has no callable 'get_action' method.")
                         done = True; break # Stop if agent cannot act
                except Exception as action_e:
                    logger.error("Error getting action from agent.", exc_info=True)
                    done = True; break

                if action is None: # Safety check
                     logger.error("Agent failed to produce an action.")
                     done = True; break

                logger.debug(f" Ep {episode}, Step {step}: Action={action}")

                # Environment steps
                try:
                    reward, step_done = game.step(action)
                    done = step_done
                    episode_reward += reward
                    step += 1
                except Exception as step_e:
                    logger.error("Error during game step.", exc_info=True)
                    done = True; break

                # Get next state if not done
                if not done:
                    try:
                        state = _get_current_state(game, agent_state_type)
                    except Exception as state_e:
                        logger.error("Error getting next state.", exc_info=True)
                        done = True; break
                else:
                    # Log final episode stats
                    final_score = getattr(game, 'score', 'N/A')
                    logger.info(f"Episode {episode} finished. Score: {final_score}, Steps: {step}, Total Reward: {episode_reward:.2f}")
                    if isinstance(final_score, (int, float)):
                        total_scores.append(final_score)
                    else:
                        total_scores.append(-1) # Assign placeholder if score is not numeric

                # Render the game state
                try:
                    game.draw()
                except Exception as draw_e:
                    logger.error("Error during game draw.", exc_info=True)
                    # Continue loop, but drawing might be broken

                # Control playback speed
                time.sleep(1.0 / max(1, target_fps)) # Prevent division by zero

            # Exit outer loop if quit attempt was made
            if quit_attempt: break

    except KeyboardInterrupt:
        logger.warning("\nPlayback interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"\nError during playback loop: {e}", exc_info=True)
    finally:
        # Log average score
        avg_score = np.mean(total_scores) if total_scores else 0
        logger.info("-" * 25)
        logger.info(f"Playback finished. Average Score over {len(total_scores)} episodes: {avg_score:.2f}")
        logger.info("Quitting Pygame...")
        if hasattr(game, 'quit') and callable(game.quit):
            game.quit() # Ensure Pygame quits cleanly
        logger.info("Playback script ended.")

# --- Main Execution Block ---
if __name__ == "__main__":
    if 'AGENT_KEYS' not in globals() or 'AGENT_CLASSES' not in globals():
        print("FATAL ERROR: Agent mappings not defined."); exit(1)

    selected_agent_key = None
    selected_agent_num = None # Keep track of number if selected via prompt

    # --- Agent Selection Logic ---
    # 1. Check command-line argument first
    if args.agent_key:
        if args.agent_key in AGENT_CLASSES and AGENT_CLASSES[args.agent_key] is not None:
            selected_agent_key = args.agent_key
            print(f"\nUsing agent specified via cmd: {selected_agent_key}")
        else:
            print(f"\nError: Invalid or unavailable agent key specified via cmd: '{args.agent_key}'")
            # selected_agent_key remains None, will trigger prompt below

    # 2. Prompt user if no valid key was provided via command line
    if selected_agent_key is None:
        print("\nAvailable Agents for Playback:")
        # Display agents with their numbers and availability status
        for num, key in sorted(AGENT_KEYS.items()):
            status = " (Unavailable)" if key not in AGENT_CLASSES or AGENT_CLASSES[key] is None else ""
            print(f"  {num}: {key}{status}")

        # Loop until a valid agent number is selected
        while selected_agent_key is None:
            try:
                num_input = input("Choose an agent number to play: ")
                num = int(num_input) # Convert input to integer

                # Check if the number corresponds to a valid and available agent
                if num in AGENT_KEYS:
                    key_check = AGENT_KEYS[num]
                    if key_check in AGENT_CLASSES and AGENT_CLASSES[key_check] is not None:
                        selected_agent_key = key_check # Store the selected agent key
                        selected_agent_num = num # Store the number too for logging
                    else:
                        print(f"Error: Agent '{key_check}' (#{num}) is unavailable.")
                else:
                    print(f"Invalid agent number '{num}'. Please choose from the list.")

            except ValueError:
                print("Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nSelection interrupted. Exiting.")
                exit() # Exit if user cancels input

    # 3. Exit if no agent was selected either way
    if selected_agent_key is None:
        print("No valid agent selected. Exiting.")
        exit(1)

    # --- Print Selection and Start Playback ---
    print(f"\nSelected Game: {args.game_key}")
    print(f"Selected Agent: {selected_agent_key}" + (f" (#{selected_agent_num})" if selected_agent_num is not None else ""))
    print(f"Target FPS: {args.fps}")
    print("-" * 30)

    play_agent(
        game_key=args.game_key,
        agent_key=selected_agent_key,
        num_episodes=args.num_episodes,
        target_fps=args.fps
    )
# --- END OF FILE project/play_trained_agent.py ---
