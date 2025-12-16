# standalone_learning_hrl_kb_agent_v24_richer_state.py
# A Proof of Concept for a HRL Agent with learning components.
# VERSION 24: Richer Symbolic State & Rewards
# - State now includes danger relative to snake's heading (forward, left, right).
# - State now includes tail direction to give the agent body awareness.
# - Intrinsic rewards are now proportional to progress with a terminal bonus.

import pygame
import random
import numpy as np
from collections import deque
import time
import logging
from typing import Union, Dict, List, Any
import os
import argparse

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses

# --- Basic Logging and Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("Learning_HRL_KB_PoC")
tf.get_logger().setLevel('ERROR')

# --- GLOBAL CONSTANTS ---
# ### SYMBOLIC STATE UPGRADE ###
# Old absolute danger features are removed.
# New relative danger and tail direction features are added.
SYMBOLIC_FEATURE_NAMES = sorted([
    'apple_dx', 'apple_dy', 'trap_dx', 'trap_dy', 'apple_dist', 'trap_dist',
    'is_apple_close', 'is_trap_close',
    'danger_forward', 'danger_left_rel', 'danger_right_rel', # NEW: Relative danger
    'tail_dx', 'tail_dy' # NEW: Tail direction
])
STATE_SIZE = len(SYMBOLIC_FEATURE_NAMES)
ACTION_SIZE = 4

# --- Environment ---
class SimpleSnakeGame:
    def __init__(self, render=True, grid_size=10):
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((grid_size * 30, grid_size * 30))
            self.font = pygame.font.SysFont('Arial', 18)
        self.render_mode = render
        self.grid_size = grid_size; self.cell_size = 30
        self.clock = pygame.time.Clock()
        self.action_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)} # N, S, W, E
        self.reset()

    def reset(self):
        # Start with a length of 2 to make tail direction meaningful from the start
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2), (self.grid_size // 2 - 1, self.grid_size // 2)])
        self.direction = (1, 0)
        self.apple_pos = self._place_item(); self.trap_pos = self._place_item(); self.obstacle_pos = self._place_item()
        self.score = 0; self.steps = 0; self.done = False
        self.steps_since_last_apple = 0
        self.starvation_limit = self.grid_size * self.grid_size
        if self.render_mode: logger.info(f"Env Reset: Player@{self.snake[0]}, Apple@{self.apple_pos}, Trap@{self.trap_pos}")
        return self.get_symbolic_state()
    
    def _place_item(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            occupied = list(self.snake)
            if hasattr(self, 'apple_pos') and self.apple_pos: occupied.append(self.apple_pos)
            if hasattr(self, 'trap_pos') and self.trap_pos: occupied.append(self.trap_pos)
            if hasattr(self, 'obstacle_pos') and self.obstacle_pos: occupied.append(self.obstacle_pos)
            if pos not in occupied: return pos
    
    def get_symbolic_state(self) -> Dict:
        state = {}; head = self.snake[0]
        
        state['apple_dx'] = self.apple_pos[0] - head[0]; state['apple_dy'] = self.apple_pos[1] - head[1]
        state['trap_dx'] = self.trap_pos[0] - head[0]; state['trap_dy'] = self.trap_pos[1] - head[1]
        state['apple_dist'] = np.linalg.norm([state['apple_dx'], state['apple_dy']])
        state['trap_dist'] = np.linalg.norm([state['trap_dx'], state['trap_dy']])
        state['is_apple_close'] = 1.0 if state['apple_dist'] < (self.grid_size / 3) else 0.0
        state['is_trap_close'] = 1.0 if state['trap_dist'] < 2.0 else 0.0
        
        # ### NEW SYMBOLIC FEATURES ###
        dx, dy = self.direction
        dir_forward = self.direction
        dir_left_rel = (dy, -dx)  # 90-degree left rotation of (dx, dy)
        dir_right_rel = (-dy, dx) # 90-degree right rotation of (dx, dy)

        state['danger_forward'] = 1.0 if self._is_danger((head[0] + dir_forward[0], head[1] + dir_forward[1])) else 0.0
        state['danger_left_rel'] = 1.0 if self._is_danger((head[0] + dir_left_rel[0], head[1] + dir_left_rel[1])) else 0.0
        state['danger_right_rel'] = 1.0 if self._is_danger((head[0] + dir_right_rel[0], head[1] + dir_right_rel[1])) else 0.0
        
        if len(self.snake) > 1:
            tail = self.snake[-1]
            state['tail_dx'] = head[0] - tail[0]
            state['tail_dy'] = head[1] - tail[1]
        else:
            state['tail_dx'] = 0; state['tail_dy'] = 0
            
        return state

    def _is_danger(self, pos):
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size): return True
        # Check against body, excluding the very last tail segment which will move away
        if pos in list(self.snake)[:-1]: return True
        if pos == self.trap_pos: return True
        if pos == self.obstacle_pos: return True
        return False
    
    def step(self, action):
        if self.done: return self.get_symbolic_state(), 0, True
        self.steps += 1; self.steps_since_last_apple += 1; reward = -0.01
        new_dir = self.action_map.get(action, self.direction)
        if len(self.snake) > 1 and new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]:
            new_dir = self.direction
        self.direction = new_dir
        head = self.snake[0]; new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        if self.steps_since_last_apple > self.starvation_limit + len(self.snake): self.done = True; reward -= 5.0
        if self._is_danger(new_head):
            self.done = True; reward = -10.0
            return self.get_symbolic_state(), reward, self.done
        self.snake.appendleft(new_head)
        if new_head == self.apple_pos:
            self.score += 1; reward = 10.0; self.apple_pos = self._place_item(); self.trap_pos = self._place_item()
            self.steps_since_last_apple = 0
        else: self.snake.pop()
        return self.get_symbolic_state(), reward, self.done

    def get_action_space_size(self) -> int: return ACTION_SIZE
    def is_done(self) -> bool: return self.done
    def render(self):
        if not self.render_mode: return
        self.screen.fill((20, 20, 20))
        for pos in self.snake: pygame.draw.rect(self.screen, (0,200,0), (pos[0]*self.cell_size, pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (200,0,0), (self.apple_pos[0]*self.cell_size, self.apple_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255,100,0), (self.trap_pos[0]*self.cell_size, self.trap_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (100,100,100), (self.obstacle_pos[0]*self.cell_size, self.obstacle_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        score_text = self.font.render(f"Score: {self.score}", True, (255,255,255)); self.screen.blit(score_text, (5, 5))
        pygame.display.flip(); self.clock.tick(60)
    def close(self):
        if self.render_mode: pygame.quit()

# --- Reusable Components ---
class SymbolicKnowledgeBase:
    def __init__(self):
        self.rules = []
        self._add_predefined_rules()
    def _add_predefined_rules(self):
        # ### KB RULE UPGRADE ###
        # The rule now uses the more intelligent 'danger_forward' feature.
        self.rules.append(("AppleRule", lambda s: s.get('is_apple_close') and not s.get('danger_forward'), {'type': 'option_bias', 'option_name': 'GoToApple', 'bias_value': 5.0}))
        self.rules.append(("TrapRule", lambda s: s.get('is_trap_close'), {'type': 'option_bias', 'option_name': 'AvoidTrap', 'bias_value': 10.0}))
    def reason(self, state): return [dict(consequence, reason=name) for name, cond, consequence in self.rules if cond(state)]

def build_dqn_network(input_size, output_size, hidden_units=(128, 64), name="DQN", learning_rate=0.00025):
    model = models.Sequential([layers.Input(shape=(input_size,))] + [layers.Dense(u, 'relu') for u in hidden_units] + [layers.Dense(output_size, 'linear')], name=name)
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
        options["GoToApple"] = {'name': "GoToApple", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), "GoToApple_Policy"), 'replay_buffer': deque(maxlen=2000), 'batch_size': 32, 'epsilon': 0.9, 'epsilon_decay': 0.99995, 'epsilon_min': 0.05}
        options["AvoidTrap"] = {'name': "AvoidTrap", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), "AvoidTrap_Policy"), 'replay_buffer': deque(maxlen=2000), 'batch_size': 32, 'epsilon': 0.9, 'epsilon_decay': 0.99995, 'epsilon_min': 0.05}
        return options

    def _setup_meta_controller(self):
        mc = {"option_names": list(self.options.keys()), "option_indices": {name: i for i, name in enumerate(self.options.keys())}, "meta_q_network": build_dqn_network(STATE_SIZE, len(self.options), (128,64), "MetaController_QNet"), "replay_buffer": deque(maxlen=5000), "batch_size": 16, "epsilon": 1.0, "epsilon_decay": 0.99995, "epsilon_min": 0.1, "gamma_meta": 0.99}
        return mc

    def _vectorize_state(self, symbolic_state: dict) -> np.ndarray:
        return np.array([symbolic_state.get(key, 0.0) for key in SYMBOLIC_FEATURE_NAMES], dtype=np.float32)

    def get_action(self, symbolic_state: dict) -> int:
        state_vec = self._vectorize_state(symbolic_state)
        if self.active_option:
            opt_name = self.active_option['name']; terminated = False
            if opt_name == "GoToApple": terminated = not symbolic_state['is_apple_close'] or symbolic_state['is_trap_close']
            elif opt_name == "AvoidTrap": terminated = not symbolic_state['is_trap_close']
            if terminated:
                self.meta_controller['replay_buffer'].append((self.option_start_state_vec, self.meta_controller['option_indices'][opt_name], self.option_cumulative_extrinsic_reward, state_vec, self.game.is_done()))
                self.active_option = None
        
        if self.active_option is None:
            q_values = self.meta_controller['meta_q_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0]
            for advice in self.kb.reason(symbolic_state):
                if advice['type'] == 'option_bias':
                    opt_idx = self.meta_controller['option_indices'].get(advice['option_name'])
                    if opt_idx is not None: q_values[opt_idx] += advice['bias_value']
            
            if np.random.rand() <= self.meta_controller['epsilon']: idx = random.randrange(len(self.options))
            else: idx = np.argmax(q_values)
            self.active_option = self.options[self.meta_controller['option_names'][idx]]
            self.option_start_state_vec = state_vec; self.option_cumulative_extrinsic_reward = 0.0
        
        opt = self.active_option
        if np.random.rand() <= opt['epsilon']: return random.randrange(ACTION_SIZE)
        return np.argmax(opt['policy_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0])

    def learn_components(self, prev_sym_state, action, reward, next_sym_state, done, global_step):
        if self.active_option:
            opt = self.active_option
            
            # --- NEW INTRINSIC REWARD CALCULATION ---
            intrinsic_reward = 0.0
            if opt['name'] == "GoToApple":
                # Big reward for the final step of eating the apple
                if reward > 5: # Game reward for eating apple is 10
                    intrinsic_reward = 20.0
                else: # Proportional reward for getting closer
                    progress = prev_sym_state['apple_dist'] - next_sym_state['apple_dist']
                    intrinsic_reward = progress * 5.0 # Scale the progress
            elif opt['name'] == "AvoidTrap":
                # Proportional reward for getting farther from the trap
                progress = next_sym_state['trap_dist'] - prev_sym_state['trap_dist']
                if progress > 0:
                    intrinsic_reward = progress * 2.0
                else:
                    intrinsic_reward = -5.0 # Stronger penalty for getting closer
            # --- END NEW INTRINSIC REWARD ---

            state_vec = self._vectorize_state(prev_sym_state); next_state_vec = self._vectorize_state(next_sym_state)
            opt['replay_buffer'].append((state_vec, action, intrinsic_reward, next_state_vec, done))
            self.option_cumulative_extrinsic_reward += reward

            if global_step % 4 == 0 and len(opt['replay_buffer']) >= opt['batch_size']:
                batch = random.sample(opt['replay_buffer'], opt['batch_size'])
                s, a, r, ns, d = map(np.array, zip(*batch))
                targets = r + 0.95 * np.amax(opt['policy_network'].predict(ns, verbose=0), axis=1) * (1 - d)
                target_f = opt['policy_network'].predict(s, verbose=0)
                for i, act in enumerate(a): target_f[i][act] = targets[i]
                opt['policy_network'].fit(s, target_f, epochs=1, verbose=0)
                if opt['epsilon'] > opt['epsilon_min']: opt['epsilon'] *= opt['epsilon_decay']
        
        if done:
            mc = self.meta_controller
            if len(mc['replay_buffer']) >= mc['batch_size']:
                batch = random.sample(mc['replay_buffer'], mc['batch_size'])
                s, o, r, ns, d = map(np.array, zip(*batch))
                targets = r + mc['gamma_meta'] * np.amax(mc['meta_q_network'].predict(ns, verbose=0), axis=1) * (1 - d)
                target_f = mc['meta_q_network'].predict(s, verbose=0)
                for i, opt_idx in enumerate(o): target_f[i][int(opt_idx)] = targets[i]
                mc['meta_q_network'].fit(s, target_f, epochs=1, verbose=0)
                if mc['epsilon'] > mc['epsilon_min']: mc['epsilon'] *= mc['epsilon_decay']
    
    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True)
        self.meta_controller['meta_q_network'].save(os.path.join(path, "meta_controller_q_net.keras"))
        for option_obj in self.options.values():
            option_obj['policy_network'].save(os.path.join(path, f"option_{option_obj['name']}_policy.keras"))
        logger.info("Models saved successfully.")

    def load_models(self, path="models"):
        try:
            self.meta_controller['meta_q_network'] = models.load_model(os.path.join(path, "meta_controller_q_net.keras"))
            for opt_name in self.options.keys():
                self.options[opt_name]['policy_network'] = models.load_model(os.path.join(path, f"option_{opt_name}_policy.keras"))
            logger.info("Models loaded successfully.")
        except (IOError, ValueError):
            logger.warning(f"Could not load saved models from '{path}'. Starting from scratch.")

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
            if done: break
        logger.info(f"--- Episode {episode} END --- Score: {env.score}, Steps: {env.steps}, Total Reward: {episode_reward:.2f}, MetaEps: {agent.meta_controller['epsilon']:.3f}\n")
        total_scores.append(env.score)
        if episode % args.save_every == 0 and episode > 0:
            agent.save_models()
    env.close()
    if total_scores: logger.info(f"\n--- Training Complete ---\nAverage score over {len(total_scores)} episodes: {np.mean(total_scores):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a learning HRL-KB agent.")
    parser.add_argument("-r", "--render", action="store_true", help="Enable graphical rendering.")
    parser.add_argument("-e", "--episodes", type=int, default=10000, help="Total number of episodes to train.")
    parser.add_argument("-s", "--save_every", type=int, default=100, help="Save the models every N episodes.")
    cli_args = parser.parse_args()
    try: main(cli_args)
    except Exception as e:
        logger.critical("An error occurred during execution.", exc_info=True)
        if 'pygame' in locals() and pygame.get_init(): pygame.quit()
