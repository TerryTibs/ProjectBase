# integrated_agent_and_game.py
# This script integrates the HRL_KB_Agent (v25) with the advanced SnakeGame
# environment from project/snake_game.py using an Adapter pattern.

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
from tensorflow.keras import models, layers, optimizers

# --- Basic Logging and Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("Integrated_HRL_Agent")
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################################################################
# SECTION 1: THE ADVANCED SNAKEGAME ENVIRONMENT (from project/snake_game.py)
################################################################################

class SnakeGame:
    def __init__(self, render=True):
        pygame.init()
        self.render = render
        self.GRID_WIDTH, self.GRID_HEIGHT = 800, 600
        self.GRID_SIZE = 20 # Larger grid size for better viewing
        self.GRID_COLS = self.GRID_WIDTH // self.GRID_SIZE
        self.GRID_ROWS = self.GRID_HEIGHT // self.GRID_SIZE
        self.SNAKE_SPEED_MULTIPLIER = 10 # Control game speed, lower is faster
        self.NUM_APPLES = 1
        
        self.state_size = 28 # This is from the original file, not directly used by our HRL agent
        self.action_size = 4
        
        self.screen = None
        if self.render:
            self.screen = pygame.display.set_mode((self.GRID_WIDTH, self.GRID_HEIGHT))
            pygame.display.set_caption("Advanced Snake Game")
            self.font = pygame.font.SysFont('Arial', 18)
            
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        start_x, start_y = self.GRID_COLS // 2, self.GRID_ROWS // 2
        self.snake = deque([(start_x, start_y)])
        self.direction = (1, 0)
        self.apples = []
        self._generate_apples()
        self.score = 0
        self.game_over = False
        self.steps = 0
        return None # The training loop will call the adapter's get_state

    def step(self, action: int):
        self.steps += 1
        reward = -0.01 # Small penalty for existing
        
        # --- Action Handling ---
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Up, Down, Left, Right
        new_direction = directions[action]
        
        # Prevent reversing
        if len(self.snake) > 1 and (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            pass # Keep current direction
        else:
            self.direction = new_direction
            
        # --- Move Snake ---
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # --- Check for Collisions ---
        if self._is_collision(new_head):
            self.game_over = True
            return -10.0, self.game_over, self.score # Return reward, done, score
        
        self.snake.appendleft(new_head)
        
        # --- Check for Apples ---
        if new_head in self.apples:
            self.score += 1
            reward = 10.0
            self.apples.remove(new_head)
            self._generate_apples()
        else:
            self.snake.pop() # Remove tail if no apple eaten
        
        if self.steps > 100 + len(self.snake) * 50: # Starvation limit
            self.game_over = True
            reward = -5.0
            
        return reward, self.game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None: pt = self.snake[0]
        # Wall collision
        if pt[0] < 0 or pt[0] >= self.GRID_COLS or pt[1] < 0 or pt[1] >= self.GRID_ROWS:
            return True
        # Self collision
        if pt in list(self.snake)[1:]:
            return True
        return False

    def _generate_apples(self):
        while len(self.apples) < self.NUM_APPLES:
            pos = (random.randint(0, self.GRID_COLS - 1), random.randint(0, self.GRID_ROWS - 1))
            if pos not in self.snake and pos not in self.apples:
                self.apples.append(pos)

    def draw(self):
        if not self.render: return
        self.screen.fill((0, 0, 0))
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
        for x, y in self.apples:
            pygame.draw.rect(self.screen, (255, 0, 0), (x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
        score_text = self.font.render(f"Score: {self.score}", True, (255,255,255)); self.screen.blit(score_text, (5, 5))
        pygame.display.flip()
        self.clock.tick(self.SNAKE_SPEED_MULTIPLIER)

    def quit(self):
        pygame.quit()

################################################################################
# SECTION 2: THE ADAPTER (TRANSLATOR)
################################################################################

class SnakeGameAdapter:
    def __init__(self, snake_game_instance: SnakeGame):
        self.game = snake_game_instance
        self.grid_size = self.game.GRID_COLS # Use width for consistency
    
    def reset(self):
        self.game.reset()
        return self.get_symbolic_state()

    def step(self, action):
        reward, done, _ = self.game.step(action)
        return self.get_symbolic_state(), reward, done

    def render(self):
        self.game.draw()

    def close(self):
        self.game.quit()
        
    def is_done(self):
        return self.game.game_over

    def get_symbolic_state(self) -> Dict:
        state = {}
        head = self.game.snake[0]

        # --- Apple Features (find closest apple) ---
        if not self.game.apples: # Handle case where apple is being generated
            apple_pos = head
        else:
            apple_pos = min(self.game.apples, key=lambda pos: (head[0] - pos[0])**2 + (head[1] - pos[1])**2)
        
        state['apple_dx'] = apple_pos[0] - head[0]
        state['apple_dy'] = apple_pos[1] - head[1]
        state['apple_dist'] = np.linalg.norm([state['apple_dx'], state['apple_dy']])
        state['is_apple_close'] = 1.0 if state['apple_dist'] < (self.grid_size / 3) else 0.0

        # --- Trap Features (Faked, as SnakeGame has no traps) ---
        # Place a fake trap very far away to make it irrelevant
        state['trap_dx'] = 999
        state['trap_dy'] = 999
        state['trap_dist'] = 999
        state['is_trap_close'] = 0.0

        # --- Relative Danger ---
        dx, dy = self.game.direction
        dir_forward = self.game.direction
        dir_left_rel = (dy, -dx)
        dir_right_rel = (-dy, dx)
        
        state['danger_forward'] = 1.0 if self.game._is_collision((head[0] + dir_forward[0], head[1] + dir_forward[1])) else 0.0
        state['danger_left_rel'] = 1.0 if self.game._is_collision((head[0] + dir_left_rel[0], head[1] + dir_left_rel[1])) else 0.0
        state['danger_right_rel'] = 1.0 if self.game._is_collision((head[0] + dir_right_rel[0], head[1] + dir_right_rel[1])) else 0.0

        # --- Tail Direction ---
        if len(self.game.snake) > 1:
            tail = self.game.snake[-1]
            state['tail_dx'] = head[0] - tail[0]
            state['tail_dy'] = head[1] - tail[1]
        else:
            state['tail_dx'] = 0
            state['tail_dy'] = 0
            
        return state

################################################################################
# SECTION 3: THE HRL AGENT AND ITS COMPONENTS (from v25)
################################################################################

SYMBOLIC_FEATURE_NAMES = sorted([
    'apple_dx', 'apple_dy', 'trap_dx', 'trap_dy', 'apple_dist', 'trap_dist',
    'is_apple_close', 'is_trap_close', 'danger_forward', 'danger_left_rel', 'danger_right_rel',
    'tail_dx', 'tail_dy'
])
STATE_SIZE = len(SYMBOLIC_FEATURE_NAMES)
ACTION_SIZE = 4

class GlobalBlackboard:
    def __init__(self): self.data = {}
    def post(self, key, value, source): logger.debug(f"BB POST by {source}: {key} = {value}")
    def read(self, key): return self.data.get(key, {}).get('value')

class SymbolicKnowledgeBase:
    def __init__(self):
        self.rules = [
            ("AppleRule", lambda s: s.get('is_apple_close') and not s.get('danger_forward'), {'type': 'option_bias', 'option_name': 'GoToApple', 'bias_value': 5.0}),
            ("TrapRule", lambda s: s.get('is_trap_close'), {'type': 'option_bias', 'option_name': 'AvoidTrap', 'bias_value': 10.0})
        ]
    def reason(self, state): return [dict(con, reason=n) for n, c, con in self.rules if c(state)]

def build_dqn_network(input_size, output_size, hidden_units=(128, 64), name="DQN", learning_rate=0.00025):
    model = models.Sequential([layers.Input(shape=(input_size,))] + [layers.Dense(u, 'relu') for u in hidden_units] + [layers.Dense(output_size, 'linear')], name=name)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

class HRL_KB_Agent:
    def __init__(self, game_adapter: SnakeGameAdapter, blackboard: GlobalBlackboard):
        self.game = game_adapter
        self.blackboard = blackboard
        self.kb = SymbolicKnowledgeBase()
        self.options = self._setup_options()
        self.meta_controller = self._setup_meta_controller()
        self.active_option: Union[Dict, None] = None
        self.option_start_state_vec = None
        self.option_cumulative_extrinsic_reward = 0.0

    def _setup_options(self):
        options = {}
        options["GoToApple"] = {'name': "GoToApple", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), "GoToApple_Policy"), 'replay_buffer': deque(maxlen=2000), 'batch_size': 32, 'epsilon': 0.9, 'epsilon_decay': 0.995, 'epsilon_min': 0.05}
        options["AvoidTrap"] = {'name': "AvoidTrap", 'policy_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), "AvoidTrap_Policy"), 'replay_buffer': deque(maxlen=2000), 'batch_size': 32, 'epsilon': 0.9, 'epsilon_decay': 0.995, 'epsilon_min': 0.05}
        return options

    def _setup_meta_controller(self):
        return {"option_names": list(self.options.keys()), "option_indices": {name: i for i, name in enumerate(self.options.keys())}, "meta_q_network": build_dqn_network(STATE_SIZE, len(self.options), (128,64), "MetaController_QNet"), "replay_buffer": deque(maxlen=5000), "batch_size": 16, "epsilon": 1.0, "epsilon_decay": 0.999, "epsilon_min": 0.1, "gamma_meta": 0.99}

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
                    if (opt_idx := self.meta_controller['option_indices'].get(advice['option_name'])) is not None:
                        q_values[opt_idx] += advice['bias_value']
            
            if np.random.rand() <= self.meta_controller['epsilon']: idx = random.randrange(len(self.options))
            else: idx = np.argmax(q_values)
            self.active_option = self.options[self.meta_controller['option_names'][idx]]
            self.blackboard.post("active_option_name", self.active_option['name'], "HRL_Agent_Meta")
            self.option_start_state_vec = state_vec; self.option_cumulative_extrinsic_reward = 0.0
        
        opt = self.active_option
        if np.random.rand() <= opt['epsilon']: return random.randrange(ACTION_SIZE)
        return np.argmax(opt['policy_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0])

    def learn_components(self, prev_sym_state, action, reward, next_sym_state, done, global_step):
        if self.active_option:
            opt = self.active_option; intrinsic_reward = 0.0
            if opt['name'] == "GoToApple":
                intrinsic_reward = 20.0 if reward > 5 else (prev_sym_state['apple_dist'] - next_sym_state['apple_dist']) * 5.0
            elif opt['name'] == "AvoidTrap":
                intrinsic_reward = (next_sym_state['trap_dist'] - prev_sym_state['trap_dist']) * 2.0
                if intrinsic_reward < 0: intrinsic_reward *= 2.5
            
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
        for opt_name, option_data in self.options.items():
            option_data['policy_network'].save(os.path.join(path, f"option_{opt_name}_policy.keras"))
        logger.info("Models saved successfully.")

    def load_models(self, path="models"):
        try:
            self.meta_controller['meta_q_network'] = models.load_model(os.path.join(path, "meta_controller_q_net.keras"))
            for opt_name in self.options.keys():
                self.options[opt_name]['policy_network'] = models.load_model(os.path.join(path, f"option_{opt_name}_policy.keras"))
            logger.info("Models loaded successfully.")
        except (IOError, ValueError):
            logger.warning(f"Could not load saved models from '{path}'. Starting fresh training.")


################################################################################
# SECTION 4: THE MAIN RUNNER
################################################################################

def main(args):
    # 1. Initialize the REAL game environment
    real_game_env = SnakeGame(render=args.render)
    
    # 2. Wrap it in our adapter
    env_adapter = SnakeGameAdapter(real_game_env)
    
    # 3. Initialize the agent, blackboard, etc.
    blackboard = GlobalBlackboard()
    agent = HRL_KB_Agent(env_adapter, blackboard)
    agent.load_models()
    
    total_scores, total_steps = [], 0
    
    for episode in range(1, args.episodes + 1):
        symbolic_state = env_adapter.reset()
        agent.active_option = None
        episode_reward, done = 0, False
        
        while not done:
            if args.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env_adapter.close()
                        return
            
            action = agent.get_action(symbolic_state)
            next_symbolic_state, reward, done = env_adapter.step(action)
            total_steps += 1
            
            agent.learn_components(symbolic_state, action, reward, next_symbolic_state, done, total_steps)
            
            episode_reward += reward
            symbolic_state = next_symbolic_state
            
            if args.render:
                env_adapter.render()
        
        logger.info(f"--- Episode {episode} END --- Score: {real_game_env.score}, Steps: {real_game_env.steps}, Total Reward: {episode_reward:.2f}\n")
        total_scores.append(real_game_env.score)

        if episode % args.save_every == 0 and episode > 0:
            agent.save_models()

    env_adapter.close()
    if total_scores: logger.info(f"\n--- Training Complete ---\nAverage score over {len(total_scores)} episodes: {np.mean(total_scores):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an integrated HRL-KB agent on the advanced SnakeGame.")
    parser.add_argument("-r", "--render", action="store_true", help="Enable graphical rendering.")
    parser.add_argument("-e", "--episodes", type=int, default=1000, help="Total number of episodes to train.")
    parser.add_argument("-s", "--save_every", type=int, default=100, help="Save the models every N episodes.")
    cli_args = parser.parse_args()
    main(cli_args)
