# standalone_learning_hrl_kb_agent_v24_richer_state.py
#
# VERSION 27.2 (This version): Corrected TypeError in Learning Logic
# - Fixed a TypeError that occurred when checking the size of the meta-controller's
#   replay buffer by correctly applying the len() function.
# - Ensured TD-error calculations are vectorized for efficiency.

import pygame
import random
import numpy as np
from collections import deque
import time
import logging
from typing import Union, Dict, Any
import os
import argparse

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

# --- Basic Logging and Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("Learning_HRL_KB_PoC")
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- GLOBAL CONSTANTS ---
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

# --- NEW COMPONENT: NoisyDense Layer for Exploration ---
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

# --- NEW COMPONENT: Prioritized Experience Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.0001):
        self.capacity = capacity
        self.alpha = alpha; self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0; self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, experience):
        if len(self.buffer) < self.capacity: self.buffer.append(experience)
        else: self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return experiences, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, td_errors, epsilon=1e-5):
        priorities = np.abs(td_errors) + epsilon
        self.priorities[batch_indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))

    def __len__(self): return len(self.buffer)

# --- Environment (Unchanged) ---
class SimpleSnakeGame:
    def __init__(self, render=True, grid_size=10):
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((grid_size * 30, grid_size * 30))
            self.font = pygame.font.SysFont('Arial', 18)
        self.render_mode = render
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
        if pos == self.trap_pos: return True;
        if pos == self.obstacle_pos: return True
        return False

    def step(self, action):
        if self.done: return self.get_symbolic_state(), 0, True
        self.steps += 1; self.steps_since_last_apple += 1; reward = -0.01
        new_dir = self.action_map.get(action, self.direction)
        if len(self.snake) > 1 and new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]: new_dir = self.direction
        self.direction = new_dir
        head = self.snake[0]; new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        if self.steps_since_last_apple > self.starvation_limit + len(self.snake): self.done = True; reward -= 5.0
        if self._is_danger(new_head): self.done = True; reward = -10.0; return self.get_symbolic_state(), reward, self.done
        self.snake.appendleft(new_head)
        if new_head == self.apple_pos:
            self.score += 1; reward = 10.0; self.apple_pos = self._place_item(); self.apple_color_name = random.choice(self.apple_color_names)
            self.trap_pos = self._place_item(); self.trap_color_name = random.choice(self.trap_color_names)
            self.steps_since_last_apple = 0
        else: self.snake.pop()
        return self.get_symbolic_state(), reward, self.done

    def get_action_space_size(self): return ACTION_SIZE
    def is_done(self): return self.done
    def render(self):
        if not self.render_mode: return
        self.screen.fill((20, 20, 20))
        for pos in self.snake: pygame.draw.rect(self.screen, (0,200,0), (pos[0]*self.cell_size, pos[1]*self.cell_size, self.cell_size, self.cell_size))
        apple_rgb = self.apple_colors[self.apple_color_name]; trap_rgb = self.trap_colors[self.trap_color_name]
        pygame.draw.rect(self.screen, apple_rgb, (self.apple_pos[0]*self.cell_size, self.apple_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, trap_rgb, (self.trap_pos[0]*self.cell_size, self.trap_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (100,100,100), (self.obstacle_pos[0]*self.cell_size, self.obstacle_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        score_text = self.font.render(f"Score: {self.score}", True, (255,255,255)); self.screen.blit(score_text, (5, 5))
        pygame.display.flip(); self.clock.tick(60)
    def close(self):
        if self.render_mode: pygame.quit()

# --- Reusable Components (KB Unchanged) ---
class SymbolicKnowledgeBase:
    def __init__(self):
        self.rules = [
            ("AppleRule", lambda s: s.get('is_apple_close') and not s.get('danger_forward'), {'type': 'option_bias', 'option_name': 'GoToApple', 'bias_value': 5.0}),
            ("TrapRule", lambda s: s.get('is_trap_close'), {'type': 'option_bias', 'option_name': 'AvoidTrap', 'bias_value': 10.0})
        ]
    def reason(self, state): return [dict(con, reason=n) for n, c, con in self.rules if c(state)]

# --- Network builder (Corrected) ---
def build_dqn_network(input_size, output_size, hidden_units=(128, 64), name="DQN", learning_rate=0.00025, use_noisy=True):
    model = models.Sequential(name=name)
    model.add(layers.Input(shape=(input_size,)))
    for units in hidden_units:
        if use_noisy:
            model.add(NoisyDense(units))
            model.add(layers.Activation('relu'))
        else:
            model.add(layers.Dense(units, activation='relu'))
    if use_noisy:
        model.add(NoisyDense(output_size))
    else:
        model.add(layers.Dense(output_size, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# --- HRL Agent Components ---
class HRL_KB_Agent:
    def __init__(self, game: SimpleSnakeGame):
        self.game = game
        self.kb = SymbolicKnowledgeBase()
        self.options = self._setup_options()
        self.meta_controller = self._setup_meta_controller()
        self.active_option: Union[Dict, None] = None
        self.option_start_state_vec = None
        self.option_cumulative_extrinsic_reward = 0.0

    def _setup_options(self):
        options = {}
        options["GoToApple"] = {'name': "GoToApple", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), "GoToApple_Policy"), 'replay_buffer': PrioritizedReplayBuffer(2000), 'batch_size': 32}
        options["AvoidTrap"] = {'name': "AvoidTrap", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), "AvoidTrap_Policy"), 'replay_buffer': PrioritizedReplayBuffer(2000), 'batch_size': 32}
        return options

    def _setup_meta_controller(self):
        return {"option_names": list(self.options.keys()), "option_indices": {name: i for i, name in enumerate(self.options.keys())}, "meta_q_network": build_dqn_network(STATE_SIZE, len(self.options), (128,64), "MetaController_QNet"), "replay_buffer": PrioritizedReplayBuffer(5000), "batch_size": 16, "gamma_meta": 0.99}

    def _vectorize_state(self, symbolic_state: dict) -> np.ndarray:
        return np.array([symbolic_state.get(key, 0.0) for key in SYMBOLIC_FEATURE_NAMES], dtype=np.float32)

    def get_action(self, symbolic_state: dict) -> int:
        state_vec = self._vectorize_state(symbolic_state)
        if self.active_option:
            opt_name = self.active_option['name']
            terminated = (opt_name == "GoToApple" and (not symbolic_state['is_apple_close'] or symbolic_state['is_trap_close'])) or (opt_name == "AvoidTrap" and not symbolic_state['is_trap_close'])
            if terminated:
                exp = (self.option_start_state_vec, self.meta_controller['option_indices'][opt_name], self.option_cumulative_extrinsic_reward, state_vec, self.game.is_done())
                self.meta_controller['replay_buffer'].add(exp)
                self.active_option = None
        if self.active_option is None:
            q_values = self.meta_controller['meta_q_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0]
            for advice in self.kb.reason(symbolic_state):
                if advice['type'] == 'option_bias':
                    opt_idx = self.meta_controller['option_indices'].get(advice['option_name'])
                    if opt_idx is not None: q_values[opt_idx] += advice['bias_value']
            idx = np.argmax(q_values)
            self.active_option = self.options[self.meta_controller['option_names'][idx]]
            self.option_start_state_vec = state_vec; self.option_cumulative_extrinsic_reward = 0.0
        opt = self.active_option
        return np.argmax(opt['policy_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0])

    def learn_components(self, prev_sym_state, action, reward, next_sym_state, done, global_step):
        if self.active_option:
            opt = self.active_option
            intrinsic_reward = 0.0
            if opt['name'] == "GoToApple": intrinsic_reward = 20.0 if reward > 5 else (prev_sym_state['apple_dist'] - next_sym_state['apple_dist']) * 5.0
            elif opt['name'] == "AvoidTrap": progress = next_sym_state['trap_dist'] - prev_sym_state['trap_dist']; intrinsic_reward = progress * 2.0 if progress > 0 else -5.0
            state_vec = self._vectorize_state(prev_sym_state); next_state_vec = self._vectorize_state(next_sym_state)
            opt['replay_buffer'].add((state_vec, action, intrinsic_reward, next_state_vec, done))
            self.option_cumulative_extrinsic_reward += reward
            if global_step % 4 == 0 and len(opt['replay_buffer']) >= opt['batch_size']:
                experiences, indices, is_weights = opt['replay_buffer'].sample(opt['batch_size'])
                s, a, r, ns, d = map(np.array, zip(*experiences))
                q_values_next = opt['policy_network'].predict(ns, verbose=0)
                targets = r + 0.95 * np.amax(q_values_next, axis=1) * (1 - d)
                target_f = opt['policy_network'].predict(s, verbose=0)
                td_errors = targets - target_f[np.arange(opt['batch_size']), a]
                opt['replay_buffer'].update_priorities(indices, td_errors)
                target_f[np.arange(opt['batch_size']), a] = targets
                opt['policy_network'].train_on_batch(s, target_f, sample_weight=is_weights)
        
        # --- CORRECTED LINE ---
        if done and len((mc := self.meta_controller)['replay_buffer']) >= mc['batch_size']:
            experiences, indices, is_weights = mc['replay_buffer'].sample(mc['batch_size'])
            s, o, r, ns, d = map(np.array, zip(*experiences))
            q_values_next = mc['meta_q_network'].predict(ns, verbose=0)
            targets = r + mc['gamma_meta'] * np.amax(q_values_next, axis=1) * (1 - d)
            target_f = mc['meta_q_network'].predict(s, verbose=0)
            o = o.astype(int)
            td_errors = targets - target_f[np.arange(mc['batch_size']), o]
            mc['replay_buffer'].update_priorities(indices, td_errors)
            target_f[np.arange(mc['batch_size']), o] = targets
            mc['meta_q_network'].train_on_batch(s, target_f, sample_weight=is_weights)

    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True)
        self.meta_controller['meta_q_network'].save(os.path.join(path, "meta_controller_q_net.keras"))
        for opt in self.options.values(): opt['policy_network'].save(os.path.join(path, f"option_{opt['name']}_policy.keras"))
        logger.info("Models saved successfully.")

    def load_models(self, path="models"):
        try:
            self.meta_controller['meta_q_network'] = models.load_model(os.path.join(path, "meta_controller_q_net.keras"), custom_objects={'NoisyDense': NoisyDense})
            for opt_name in self.options.keys(): self.options[opt_name]['policy_network'] = models.load_model(os.path.join(path, f"option_{opt_name}_policy.keras"), custom_objects={'NoisyDense': NoisyDense})
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load saved models from '{path}'. This is expected if the model architecture has changed. Starting from scratch. Error: {e}")

# --- Main Runner ---
def main(args):
    env = SimpleSnakeGame(render=args.render); agent = HRL_KB_Agent(env)
    agent.load_models()
    total_scores = []; total_steps = 0
    for episode in range(1, args.episodes + 1):
        symbolic_state = env.reset()
        agent.active_option = None; episode_reward = 0; done = False
        while not done:
            if env.render_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return
            action = agent.get_action(symbolic_state)
            next_symbolic_state, reward, done = env.step(action)
            total_steps += 1
            agent.learn_components(symbolic_state, action, reward, next_symbolic_state, done, total_steps)
            episode_reward += reward; symbolic_state = next_symbolic_state
            if env.render_mode: env.render()
        logger.info(f"--- Episode {episode} END --- Score: {env.score}, Steps: {env.steps}, Total Reward: {episode_reward:.2f}")
        total_scores.append(env.score)
        if episode % args.save_every == 0 and episode > 0: agent.save_models()
    env.close()
    if total_scores: logger.info(f"\n--- Training Complete ---\nAverage score over {len(total_scores)} episodes: {np.mean(total_scores):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a learning HRL-KB agent.")
    parser.add_argument("-r", "--render", action="store_true", help="Enable graphical rendering.")
    parser.add_argument("-e", "--episodes", type=int, default=10000, help="Total number of episodes to train.")
    parser.add_argument("-s", "--save_every", type=int, default=100, help="Save the models every N episodes.")
    cli_args = parser.parse_args()
    main(cli_args)
