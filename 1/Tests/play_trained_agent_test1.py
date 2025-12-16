# play_trained_agent_v27.py
# A script to load and watch the trained HRL-KB agent play the game.
# VERSION 27: Synchronized with the latest training script.
# - Added the NoisyDense custom layer class required for model loading.
# - Updated GLOBAL_CONSTANTS and SYMBOLIC_FEATURE_NAMES to match the 25 features.
# - Replaced the entire SimpleSnakeGame class with the correct version.
# - Updated the model loading calls to include the custom_objects argument.

import pygame
import random
import numpy as np
from collections import deque
import time
import logging
from typing import Dict
import os

import tensorflow as tf
from tensorflow.keras import models, layers

# --- Basic Logging and Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("HRL_Playback")
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- REQUIRED: Definition of the custom NoisyDense layer ---
class NoisyDense(layers.Layer):
    """A dense layer with factorized Gaussian noise for exploration."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel_mu = self.add_weight(shape=(self.input_dim, self.units), initializer='he_uniform', name='kernel_mu')
        self.bias_mu = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_mu')
        sigma_init = tf.constant_initializer(0.5 / np.sqrt(self.input_dim))
        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units), initializer=sigma_init, name='kernel_sigma')
        self.bias_sigma = self.add_weight(shape=(self.units,), initializer=sigma_init, name='bias_sigma')

    def call(self, inputs):
        epsilon_in = self._f(tf.random.normal(shape=(self.input_dim, 1)))
        epsilon_out = self._f(tf.random.normal(shape=(1, self.units)))
        kernel_epsilon = tf.matmul(epsilon_in, epsilon_out)
        bias_epsilon = epsilon_out
        kernel = self.kernel_mu + self.kernel_sigma * kernel_epsilon
        bias = self.bias_mu + self.bias_sigma * tf.squeeze(bias_epsilon)
        return tf.matmul(inputs, kernel) + bias

    def _f(self, x):
        return tf.sign(x) * tf.sqrt(tf.abs(x))

# --- GLOBAL CONSTANTS (Must match the training script EXACTLY) ---
APPLE_COLORS = {'red': (200, 0, 0), 'blue': (0, 0, 200), 'green': (0, 200, 0)}
TRAP_COLORS = {'orange': (255, 100, 0), 'purple': (128, 0, 128), 'cyan': (0, 200, 200)}
APPLE_COLOR_NAMES = sorted(APPLE_COLORS.keys())
TRAP_COLOR_NAMES = sorted(TRAP_COLORS.keys())

apple_color_features = [f'apple_color_{name}' for name in APPLE_COLOR_NAMES]
trap_color_features = [f'trap_color_{name}' for name in TRAP_COLOR_NAMES]

SYMBOLIC_FEATURE_NAMES = sorted([
    'apple_dx', 'apple_dy', 'trap_dx', 'trap_dy', 'apple_dist', 'trap_dist',
    'is_apple_close', 'is_trap_close', 'danger_forward', 'danger_left_rel',
    'danger_right_rel', 'tail_dx', 'tail_dy', 'snake_length_normalized',
    'is_on_edge', 'is_apple_aligned_x', 'is_apple_aligned_y',
    'apple_vector_x', 'apple_vector_y'
] + apple_color_features + trap_color_features)

STATE_SIZE = len(SYMBOLIC_FEATURE_NAMES)
ACTION_SIZE = 4

# --- SYNCHRONIZED Environment Class ---
class SimpleSnakeGame:
    def __init__(self, render=True, grid_size=10):
        self.render_mode = render
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((grid_size * 30, grid_size * 30))
            pygame.display.set_caption("Playback of Trained HRL-KB Agent")
            self.font = pygame.font.SysFont('Arial', 18)
        self.grid_size = grid_size; self.cell_size = 30
        self.clock = pygame.time.Clock()
        self.action_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        self.apple_colors = APPLE_COLORS; self.trap_colors = TRAP_COLORS
        self.apple_color_names = APPLE_COLOR_NAMES; self.trap_color_names = TRAP_COLOR_NAMES
        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2), (self.grid_size // 2 - 1, self.grid_size // 2)])
        self.direction = (1, 0)
        self.apple_pos = self._place_item(); self.apple_color_name = random.choice(self.apple_color_names)
        self.trap_pos = self._place_item(); self.trap_color_name = random.choice(self.trap_color_names)
        self.obstacle_pos = self._place_item()
        self.score = 0; self.steps = 0; self.done = False; self.steps_since_last_apple = 0
        self.starvation_limit = self.grid_size * self.grid_size
        logger.info(f"Env Reset: Player@{self.snake[0]}, Apple@{self.apple_pos} ({self.apple_color_name}), Trap@{self.trap_pos} ({self.trap_color_name})")
        return self.get_symbolic_state()

    def _place_item(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            occupied = list(self.snake) + [getattr(self, 'apple_pos', None), getattr(self, 'trap_pos', None), getattr(self, 'obstacle_pos', None)]
            if pos not in occupied: return pos

    def get_symbolic_state(self) -> Dict:
        state = {}; head = self.snake[0]
        state['apple_dx'], state['apple_dy'] = self.apple_pos[0] - head[0], self.apple_pos[1] - head[1]
        state['trap_dx'], state['trap_dy'] = self.trap_pos[0] - head[0], self.trap_pos[1] - head[1]
        state['apple_dist'] = np.linalg.norm([state['apple_dx'], state['apple_dy']])
        state['trap_dist'] = np.linalg.norm([state['trap_dx'], state['trap_dy']])
        state['is_apple_close'] = 1.0 if state['apple_dist'] < (self.grid_size / 3) else 0.0
        state['is_trap_close'] = 1.0 if state['trap_dist'] < 2.0 else 0.0
        dx, dy = self.direction; dir_forward = self.direction; dir_left_rel = (dy, -dx); dir_right_rel = (-dy, dx)
        state['danger_forward'] = 1.0 if self._is_danger((head[0] + dir_forward[0], head[1] + dir_forward[1])) else 0.0
        state['danger_left_rel'] = 1.0 if self._is_danger((head[0] + dir_left_rel[0], head[1] + dir_left_rel[1])) else 0.0
        state['danger_right_rel'] = 1.0 if self._is_danger((head[0] + dir_right_rel[0], head[1] + dir_right_rel[1])) else 0.0
        state['tail_dx'], state['tail_dy'] = (head[0] - self.snake[-1][0], head[1] - self.snake[-1][1]) if len(self.snake) > 1 else (0, 0)
        max_len = self.grid_size * self.grid_size; state['snake_length_normalized'] = len(self.snake) / max_len if max_len > 0 else 0.0
        head_x, head_y = head; state['is_on_edge'] = 1.0 if (head_x == 0 or head_x == self.grid_size - 1 or head_y == 0 or head_y == self.grid_size - 1) else 0.0
        state['is_apple_aligned_x'], state['is_apple_aligned_y'] = (1.0 if state['apple_dx'] == 0 else 0.0, 1.0 if state['apple_dy'] == 0 else 0.0)
        state['apple_vector_x'], state['apple_vector_y'] = (state['apple_dx'] / state['apple_dist'], state['apple_dy'] / state['apple_dist']) if state['apple_dist'] > 1e-6 else (0.0, 0.0)
        for name in self.apple_color_names: state[f'apple_color_{name}'] = 1.0 if name == self.apple_color_name else 0.0
        for name in self.trap_color_names: state[f'trap_color_{name}'] = 1.0 if name == self.trap_color_name else 0.0
        return state

    def _is_danger(self, pos):
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size): return True
        if pos in list(self.snake)[:-1]: return True
        if pos == self.trap_pos: return True
        if pos == self.obstacle_pos: return True
        return False

    def step(self, action):
        if self.done: return self.get_symbolic_state(), 0, True
        self.steps += 1
        new_dir = self.action_map.get(action, self.direction)
        if len(self.snake) > 1 and new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]: new_dir = self.direction
        self.direction = new_dir
        head = self.snake[0]; new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        if self._is_danger(new_head): self.done = True; return self.get_symbolic_state(), -10.0, self.done
        self.snake.appendleft(new_head)
        if new_head == self.apple_pos:
            self.score += 1; self.apple_pos = self._place_item(); self.apple_color_name = random.choice(self.apple_color_names)
            self.trap_pos = self._place_item(); self.trap_color_name = random.choice(self.trap_color_names)
        else: self.snake.pop()
        return self.get_symbolic_state(), 0, self.done

    def render(self):
        if not self.render_mode: return
        self.screen.fill((20, 20, 20))
        for pos in self.snake: pygame.draw.rect(self.screen, (0,200,0), (pos[0]*self.cell_size, pos[1]*self.cell_size, self.cell_size, self.cell_size))
        apple_rgb = self.apple_colors[self.apple_color_name]; trap_rgb = self.trap_colors[self.trap_color_name]
        pygame.draw.rect(self.screen, apple_rgb, (self.apple_pos[0]*self.cell_size, self.apple_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, trap_rgb, (self.trap_pos[0]*self.cell_size, self.trap_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (100,100,100), (self.obstacle_pos[0]*self.cell_size, self.obstacle_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        score_text = self.font.render(f"Score: {self.score}", True, (255,255,255)); self.screen.blit(score_text, (5, 5))
        pygame.display.flip(); self.clock.tick(10) # Slowed down for nice viewing
    def close(self):
        if self.render_mode: pygame.quit()

# --- Agent for Playback ---
class HRL_Playback_Agent:
    def __init__(self, game):
        self.game = game
        self.options = {}
        self.meta_controller = {}
        self.active_option = None
        self.load_models()

    def _vectorize_state(self, symbolic_state: dict) -> np.ndarray:
        return np.array([symbolic_state.get(key, 0.0) for key in SYMBOLIC_FEATURE_NAMES], dtype=np.float32)

    def load_models(self, path="models"):
        logger.info(f"Attempting to load trained models from '{path}' directory...")
        try:
            # --- UPDATED: Added custom_objects argument ---
            custom_obj = {'NoisyDense': NoisyDense}
            self.meta_controller['meta_q_network'] = models.load_model(os.path.join(path, "meta_controller_q_net.keras"), custom_objects=custom_obj)
            self.meta_controller['option_names'] = ["GoToApple", "AvoidTrap"]
            self.meta_controller['option_indices'] = {"GoToApple": 0, "AvoidTrap": 1}
            self.options["GoToApple"] = {'name': "GoToApple", 'policy_network': models.load_model(os.path.join(path, "option_GoToApple_policy.keras"), custom_objects=custom_obj)}
            self.options["AvoidTrap"] = {'name': "AvoidTrap", 'policy_network': models.load_model(os.path.join(path, "option_AvoidTrap_policy.keras"), custom_objects=custom_obj)}
            logger.info("All models loaded successfully.")
        except (IOError, ValueError) as e:
            logger.critical(f"FATAL: Could not load saved models. Make sure you have trained the agent first.")
            logger.critical(f"Reason: {e}")
            raise SystemExit("Required model files not found.")

    def get_action(self, symbolic_state: dict) -> int:
        state_vec = self._vectorize_state(symbolic_state)
        if self.active_option:
            opt_name = self.active_option['name']
            terminated = (opt_name == "GoToApple" and (not symbolic_state['is_apple_close'] or symbolic_state['is_trap_close'])) or \
                         (opt_name == "AvoidTrap" and not symbolic_state['is_trap_close'])
            if terminated:
                logger.info(f"Option '{opt_name}' terminated.")
                self.active_option = None
        
        if self.active_option is None:
            q_values = self.meta_controller['meta_q_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0]
            best_option_idx = np.argmax(q_values)
            self.active_option = self.options[self.meta_controller['option_names'][best_option_idx]]
            logger.info(f"Meta Selected: '{self.active_option['name']}' with Qs: {[round(q, 2) for q in q_values]}")
        
        opt = self.active_option
        q_values_opt = opt['policy_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0]
        return np.argmax(q_values_opt)

# --- Main Playback Runner ---
def main():
    logger.info("--- Initializing Agent PLAYBACK ---")
    
    try:
        env = SimpleSnakeGame(render=True)
        agent = HRL_Playback_Agent(env)
    except SystemExit as e:
        logger.error(e)
        return
    
    num_episodes = 10
    total_scores = []
    
    for episode in range(1, num_episodes + 1):
        logger.info(f"\n{'='*20} Playback Episode {episode}/{num_episodes} {'='*20}")
        symbolic_state = env.reset()
        agent.active_option = None
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.warning("Quit command received. Shutting down.")
                    env.close()
                    return
            
            action = agent.get_action(symbolic_state)
            
            active_opt_name = agent.active_option['name'] if agent.active_option else 'None'
            
            # This check is just to make the log output a bit cleaner
            action_map_str = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
            action_str = action_map_str.get(action, 'Unknown')
            
            logger.info(f"Step {env.steps + 1:<4} | Active Option: {active_opt_name:<15} | Chosen Action: {action_str}")

            next_symbolic_state, _, done = env.step(action)
            
            symbolic_state = next_symbolic_state
            env.render()
            
        logger.info(f"--- Episode {episode} END --- Final Score: {env.score}\n")
        total_scores.append(env.score)

    env.close()
    if total_scores:
        logger.info(f"\n--- Playback Complete ---")
        logger.info(f"Average score over {len(total_scores)} episodes: {np.mean(total_scores):.2f}")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        logger.critical("An error occurred during execution.", exc_info=True)
        if 'pygame' in locals() and pygame.get_init(): pygame.quit()
