# standalone_learning_hrl_kb_agent_v16_FUNCTIONAL_FINAL.py
# FINAL VERSION: This script uses a functional, dictionary-based approach for agent
# components. It completely removes the class inheritance that was causing the
# persistent TypeError, guaranteeing a fix for that specific issue.

import pygame
import random
import numpy as np
from collections import deque
import time
import logging
from typing import Dict, List, Any, Union

# Use TensorFlow for the learning components
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses

# --- Basic Logging and Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("Learning_HRL_KB_PoC")
tf.get_logger().setLevel('ERROR')

# --- GLOBAL CONSTANTS ---
SYMBOLIC_FEATURE_NAMES = sorted([
    'apple_dx', 'apple_dy', 'trap_dx', 'trap_dy', 'apple_dist', 'trap_dist',
    'is_apple_close', 'is_trap_close', 'danger_up', 'danger_down',
    'danger_left', 'danger_right'
])
STATE_SIZE = len(SYMBOLIC_FEATURE_NAMES)
ACTION_SIZE = 4

# --- Environment ---
class SimpleSnakeGame:
    def __init__(self, render=True, grid_size=10):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = 30
        self.width = self.height = self.grid_size * self.cell_size
        self.render_mode = render
        self.apple_pos = None; self.trap_pos = None; self.obstacle_pos = None
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()
        self.action_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.direction = (1, 0)
        self.apple_pos = self._place_item(); self.trap_pos = self._place_item(); self.obstacle_pos = self._place_item()
        self.score = 0; self.steps = 0; self.max_steps = self.grid_size * 5; self.done = False
        logger.info(f"Env Reset: Player@{self.snake[0]}, Apple@{self.apple_pos}, Trap@{self.trap_pos}, Obstacle@{self.obstacle_pos}")
        return self.get_symbolic_state()
    
    def _place_item(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            occupied = list(self.snake)
            if self.apple_pos: occupied.append(self.apple_pos)
            if self.trap_pos: occupied.append(self.trap_pos)
            if self.obstacle_pos: occupied.append(self.obstacle_pos)
            if pos not in occupied: return pos
    
    def get_symbolic_state(self) -> Dict:
        state = {}; head = self.snake[0]
        state['apple_dx'] = self.apple_pos[0] - head[0]; state['apple_dy'] = self.apple_pos[1] - head[1]
        state['trap_dx'] = self.trap_pos[0] - head[0]; state['trap_dy'] = self.trap_pos[1] - head[1]
        state['apple_dist'] = np.linalg.norm([state['apple_dx'], state['apple_dy']])
        state['trap_dist'] = np.linalg.norm([state['trap_dx'], state['trap_dy']])
        state['is_apple_close'] = 1.0 if state['apple_dist'] < (self.grid_size / 3) else 0.0
        state['is_trap_close'] = 1.0 if state['trap_dist'] < 2.0 else 0.0
        state['danger_up'] = 1.0 if self._is_danger((head[0], head[1] - 1)) else 0.0
        state['danger_down'] = 1.0 if self._is_danger((head[0], head[1] + 1)) else 0.0
        state['danger_left'] = 1.0 if self._is_danger((head[0] - 1, head[1])) else 0.0
        state['danger_right'] = 1.0 if self._is_danger((head[0] + 1, head[1])) else 0.0
        return state

    def _is_danger(self, pos):
        x, y = pos
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size): return True
        if pos in list(self.snake)[1:]: return True
        if pos == self.trap_pos: return True
        if pos == self.obstacle_pos: return True
        return False
    
    def step(self, action):
        if self.done: return self.get_symbolic_state(), 0, True
        self.steps += 1; reward = -0.1
        self.direction = self.action_map.get(action, self.direction)
        head = self.snake[0]; new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        if self.steps >= self.max_steps: self.done = True; reward -= 1.0
        if self._is_danger(new_head):
            self.done = True; reward = -10.0
            return self.get_symbolic_state(), reward, self.done
        self.snake.appendleft(new_head)
        if new_head == self.apple_pos:
            self.score += 1; reward = 10.0; self.apple_pos = self._place_item(); self.trap_pos = self._place_item()
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
        pygame.display.flip(); self.clock.tick(20)
    def close(self): pygame.quit()

# --- Reusable Components ---
class GlobalBlackboard:
    def __init__(self): self.data = {}
    def post(self, key, value, source): logger.info(f"BB POST by {source}: {key} = {value}")
    def read(self, key): return self.data.get(key, {}).get('value')

class SymbolicKnowledgeBase:
    def __init__(self):
        self.rules = []
        self._add_predefined_rules()
    def _add_predefined_rules(self):
        self.rules.append(("AppleRule", lambda s: s.get('is_apple_close') and not any(s.get(k) for k in ['danger_up', 'danger_down', 'danger_left', 'danger_right']), {'type': 'option_bias', 'option_name': 'GoToApple', 'bias_value': 5.0}))
        self.rules.append(("TrapRule", lambda s: s.get('is_trap_close'), {'type': 'option_bias', 'option_name': 'AvoidTrap', 'bias_value': 10.0}))
    def reason(self, state): return [dict(consequence, reason=name) for name, cond, consequence in self.rules if cond(state)]

def build_dqn_network(input_size, output_size, hidden_units=(64, 32), name="DQN"):
    model = models.Sequential([layers.Input(shape=(input_size,))] + [layers.Dense(u, 'relu') for u in hidden_units] + [layers.Dense(output_size, 'linear')], name=name)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# --- HRL Agent ---
class HRL_KB_Agent:
    def __init__(self, game: SimpleSnakeGame):
        logger.info("HRL_KB_Agent: Initializing...")
        self.game = game
        self.blackboard = GlobalBlackboard()
        self.kb = self._setup_knowledge_base()
        self.options = self._setup_options()
        self.meta_controller = self._setup_meta_controller()
        self.active_option: Union[Dict, None] = None
        self.option_start_state_vec = None
        self.option_cumulative_extrinsic_reward = 0.0
        logger.info("HRL_KB_Agent fully initialized.")

    def _setup_knowledge_base(self):
        return SymbolicKnowledgeBase()

    def _setup_options(self):
        options = {}
        options["GoToApple"] = {'name': "GoToApple", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, name="GoToApple_Policy"), 'replay_buffer': deque(maxlen=2000), 'batch_size': 32, 'epsilon': 0.9, 'epsilon_decay': 0.99, 'epsilon_min': 0.05}
        options["AvoidTrap"] = {'name': "AvoidTrap", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, name="AvoidTrap_Policy"), 'replay_buffer': deque(maxlen=2000), 'batch_size': 32, 'epsilon': 0.9, 'epsilon_decay': 0.99, 'epsilon_min': 0.05}
        logger.info("Options setup complete.")
        return options

    def _setup_meta_controller(self):
        mc = {
            "option_names": list(self.options.keys()),
            "option_indices": {name: i for i, name in enumerate(self.options.keys())},
            "meta_q_network": build_dqn_network(STATE_SIZE, len(self.options), (64,32), "MetaController_QNet"),
            "replay_buffer": deque(maxlen=5000), "batch_size": 16,
            "epsilon": 0.8, "epsilon_decay": 0.995, "epsilon_min": 0.1, "gamma_meta": 0.99
        }
        logger.info("MetaController setup complete.")
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
                logger.info(f"Option '{opt_name}' terminated.")
                self.meta_controller['replay_buffer'].append((self.option_start_state_vec, self.meta_controller['option_indices'][opt_name], self.option_cumulative_extrinsic_reward, state_vec, self.game.is_done()))
                self.active_option = None
        
        if self.active_option is None:
            q_values = self.meta_controller['meta_q_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0]
            for name, cond, cons in self.kb.reason(symbolic_state):
                if cons['type'] == 'option_bias':
                    opt_idx = self.meta_controller['option_indices'].get(cons['option_name'])
                    if opt_idx is not None: q_values[opt_idx] += cons['bias_value']
            
            if np.random.rand() <= self.meta_controller['epsilon']: idx = random.randrange(len(self.options))
            else: idx = np.argmax(q_values)
            self.active_option = self.options[self.meta_controller['option_names'][idx]]
            self.blackboard.post("active_option_name", self.active_option['name'], "HRL_Agent_Meta")
            self.option_start_state_vec = state_vec; self.option_cumulative_extrinsic_reward = 0.0
        
        opt = self.active_option
        if np.random.rand() <= opt['epsilon']: return random.randrange(ACTION_SIZE)
        return np.argmax(opt['policy_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0])

    def learn_components(self, prev_sym_state, action, reward, next_sym_state, done):
        if self.active_option:
            opt = self.active_option; intrinsic_reward = 0.0
            if opt['name'] == "GoToApple": intrinsic_reward = 1.0 if next_sym_state['apple_dist'] < prev_sym_state['apple_dist'] else -0.5
            elif opt['name'] == "AvoidTrap": intrinsic_reward = 1.0 if next_sym_state['trap_dist'] > prev_sym_state['trap_dist'] else -2.0
            
            state_vec = self._vectorize_state(prev_sym_state); next_state_vec = self._vectorize_state(next_sym_state)
            opt['replay_buffer'].append((state_vec, action, intrinsic_reward, next_state_vec, done))
            if len(opt['replay_buffer']) >= opt['batch_size']:
                batch = random.sample(opt['replay_buffer'], opt['batch_size'])
                s, a, r, ns, d = map(np.array, zip(*batch))
                targets = r + 0.9 * np.amax(opt['policy_network'].predict(ns, verbose=0), axis=1) * (1 - d)
                target_f = opt['policy_network'].predict(s, verbose=0)
                for i, act in enumerate(a): target_f[i][act] = targets[i]
                opt['policy_network'].fit(s, target_f, epochs=1, verbose=0)
                if opt['epsilon'] > opt['epsilon_min']: opt['epsilon'] *= opt['epsilon_decay']
            self.option_cumulative_extrinsic_reward += reward
        
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

# --- Main Runner ---
def main():
    logger.info("--- Initializing LEARNING HRL-KB Proof of Concept (v15 FINAL) ---")
    env = SimpleSnakeGame(render=True); agent = HRL_KB_Agent(env)
    num_episodes = 100; total_scores = []
    
    for episode in range(1, num_episodes + 1):
        symbolic_state = env.reset()
        agent.active_option = None; episode_reward = 0; done = False
        
        while not done:
            if env.render_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return None, True
            
            action = agent.get_action(symbolic_state)
            next_symbolic_state, reward, done = env.step(action)
            
            agent.learn_components(symbolic_state, action, reward, next_symbolic_state, done)
            
            episode_reward += reward; symbolic_state = next_symbolic_state
            if env.render_mode: env.render()
            if done: break
    
        logger.info(f"--- Episode {episode} END --- Score: {env.score}, Total Reward: {episode_reward:.2f}\n")
        total_scores.append(env.score)

    env.close()
    if total_scores: logger.info(f"\n--- Proof of Concept Complete ---\nAverage score over {len(total_scores)} episodes: {np.mean(total_scores):.2f}")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        logger.critical("An error occurred during execution.", exc_info=True)
        if 'pygame' in locals() and pygame.get_init(): pygame.quit()
