# final_integrated_agent_v29.1.py
#
# This script represents the final integration of our most advanced agent
# with the advanced SnakeGame environment from the original user request.
#
# VERSION 29.1 FIX: Restored the missing SymbolicKnowledgeBase class definition.
#
# AGENT FEATURES:
# - Hierarchical Reinforcement Learning (HRL)
# - Double Dueling DQN Architecture
# - Prioritized Experience Replay (PER)
# - Noisy Nets for Exploration
# - Multi-step (N-step) Returns for faster learning
# - Rich 25-feature Symbolic State
#
# ENVIRONMENT:
# - The full, complex SnakeGame from project/snake_game.py

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
logger = logging.getLogger("Final_HRL_Agent")
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
        self.GRID_SIZE = 20
        self.GRID_COLS = self.GRID_WIDTH // self.GRID_SIZE
        self.GRID_ROWS = self.GRID_HEIGHT // self.GRID_SIZE
        self.SNAKE_SPEED_MULTIPLIER = 15
        self.NUM_APPLES = 20
        
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
        return None

    def step(self, action: int):
        self.steps += 1
        reward = -0.01
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        new_direction = directions[action]
        if len(self.snake) > 1 and (new_direction[0] * -1, new_direction[1] * -1) == self.direction:
            pass
        else:
            self.direction = new_direction
            
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        if self._is_collision(new_head):
            self.game_over = True
            return -10.0, self.game_over, self.score
        
        self.snake.appendleft(new_head)
        
        if new_head in self.apples:
            self.score += 1
            reward = 10.0
            self.apples.remove(new_head)
            self._generate_apples()
        else:
            self.snake.pop()
        
        if self.steps > 150 + len(self.snake) * 50:
            self.game_over = True
            reward = -5.0
            
        return reward, self.game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None: pt = self.snake[0]
        if pt[0] < 0 or pt[0] >= self.GRID_COLS or pt[1] < 0 or pt[1] >= self.GRID_ROWS: return True
        if pt in list(self.snake)[1:]: return True
        return False

    def _generate_apples(self):
        while len(self.apples) < self.NUM_APPLES:
            pos = (random.randint(0, self.GRID_COLS - 1), random.randint(0, self.GRID_ROWS - 1))
            if pos not in self.snake and pos not in self.apples: self.apples.append(pos)

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
# SECTION 2: THE ADVANCED AI COMPONENTS
################################################################################

class NoisyDense(layers.Layer):
    def __init__(self, units, **kwargs): super().__init__(**kwargs); self.units = units
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel_mu = self.add_weight(shape=(self.input_dim, self.units), initializer='he_uniform', name='kernel_mu')
        self.bias_mu = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_mu')
        sigma_init = tf.constant_initializer(0.5 / np.sqrt(self.input_dim))
        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units), initializer=sigma_init, name='kernel_sigma')
        self.bias_sigma = self.add_weight(shape=(self.units,), initializer=sigma_init, name='bias_sigma')
    def call(self, inputs):
        epsilon_in = self._f(tf.random.normal(shape=(self.input_dim, 1))); epsilon_out = self._f(tf.random.normal(shape=(1, self.units)))
        kernel_epsilon = tf.matmul(epsilon_in, epsilon_out); bias_epsilon = epsilon_out
        kernel = self.kernel_mu + self.kernel_sigma * kernel_epsilon; bias = self.bias_mu + self.bias_sigma * tf.squeeze(bias_epsilon)
        return tf.matmul(inputs, kernel) + bias
    def _f(self, x): return tf.sign(x) * tf.sqrt(tf.abs(x))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.0001):
        self.capacity=capacity; self.alpha=alpha; self.beta=beta; self.beta_increment=beta_increment; self.pos=0; self.buffer=[]; self.priorities=np.zeros((capacity,), dtype=np.float32); self.max_priority=1.0
    def add(self, experience):
        if len(self.buffer) < self.capacity: self.buffer.append(experience)
        else: self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority; self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        probs = self.priorities[:len(self.buffer)] ** self.alpha; probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        total = len(self.buffer); weights = (total * probs[indices]) ** (-self.beta); weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return experiences, indices, np.array(weights, dtype=np.float32)
    def update_priorities(self, batch_indices, td_errors, epsilon=1e-5):
        priorities = np.abs(td_errors) + epsilon; self.priorities[batch_indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))
    def __len__(self): return len(self.buffer)

# --- CLASS RESTORED ---
class SymbolicKnowledgeBase:
    """A simple rule-based system to provide advice to the agent."""
    def __init__(self):
        self.rules = [
            ("AppleRule", 
             lambda s: s.get('is_apple_close') and not s.get('danger_forward'),
             {'type': 'option_bias', 'option_name': 'GoToApple', 'bias_value': 5.0}),
            
            ("TrapRule",
             lambda s: s.get('is_trap_close'),
             {'type': 'option_bias', 'option_name': 'AvoidTrap', 'bias_value': 10.0})
        ]

    def reason(self, state):
        """Evaluates all rules against the state and returns triggered advice."""
        return [dict(consequence, reason=name) for name, cond, consequence in self.rules if cond(state)]

def build_dqn_network(input_size, output_size, hidden_units=(128, 64), name="DQN", learning_rate=0.00025, use_noisy=True, dueling=True):
    inputs = layers.Input(shape=(input_size,)); x = inputs
    for units in hidden_units:
        if use_noisy: x = NoisyDense(units)(x); x = layers.Activation('relu')(x)
        else: x = layers.Dense(units, activation='relu')(x)
    if dueling:
        value_head = NoisyDense(1, name='value_head')(x) if use_noisy else layers.Dense(1, name='value_head')(x)
        advantage_head = NoisyDense(output_size, name='advantage_head')(x) if use_noisy else layers.Dense(output_size, name='advantage_head')(x)
        def aggregate_streams(streams): value, advantage = streams; return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        outputs = layers.Lambda(aggregate_streams, name='combine_q_values')([value_head, advantage_head])
    else: outputs = NoisyDense(output_size)(x) if use_noisy else layers.Dense(output_size)(x)
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

################################################################################
# SECTION 3: THE ADAPTER
################################################################################

APPLE_COLOR_NAMES = sorted(['red', 'blue', 'green'])
TRAP_COLOR_NAMES = sorted(['orange', 'purple', 'cyan'])
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

class SnakeGameAdapter:
    def __init__(self, snake_game_instance: SnakeGame):
        self.game = snake_game_instance
    def reset(self): self.game.reset(); return self.get_symbolic_state()
    def step(self, action): reward, done, _ = self.game.step(action); return self.get_symbolic_state(), reward, done
    def render(self): self.game.draw()
    def close(self): self.game.quit()
    def is_done(self): return self.game.game_over
    def get_symbolic_state(self) -> Dict:
        state = {}; head = self.game.snake[0]; grid_cols, grid_rows = self.game.GRID_COLS, self.game.GRID_ROWS
        apple_pos = min(self.game.apples, key=lambda p: (head[0]-p[0])**2 + (head[1]-p[1])**2) if self.game.apples else head
        state['apple_dx'], state['apple_dy'] = apple_pos[0] - head[0], apple_pos[1] - head[1]
        state['apple_dist'] = np.linalg.norm([state['apple_dx'], state['apple_dy']])
        state['is_apple_close'] = 1.0 if state['apple_dist'] < (grid_cols / 3) else 0.0
        state['is_apple_aligned_x'], state['is_apple_aligned_y'] = (1.0 if state['apple_dx'] == 0 else 0.0, 1.0 if state['apple_dy'] == 0 else 0.0)
        state['apple_vector_x'], state['apple_vector_y'] = (state['apple_dx']/state['apple_dist'], state['apple_dy']/state['apple_dist']) if state['apple_dist'] > 1e-6 else (0.0,0.0)
        state['trap_dx'], state['trap_dy'], state['trap_dist'] = 999, 999, 999; state['is_trap_close'] = 0.0
        for name in APPLE_COLOR_NAMES: state[f'apple_color_{name}'] = 1.0 if name == 'red' else 0.0
        for name in TRAP_COLOR_NAMES: state[f'trap_color_{name}'] = 0.0
        dx, dy = self.game.direction
        state['danger_forward'] = 1.0 if self.game._is_collision((head[0]+dx, head[1]+dy)) else 0.0
        state['danger_left_rel'] = 1.0 if self.game._is_collision((head[0]+dy, head[1]-dx)) else 0.0
        state['danger_right_rel'] = 1.0 if self.game._is_collision((head[0]-dy, head[1]+dx)) else 0.0
        state['tail_dx'], state['tail_dy'] = (head[0] - self.game.snake[-1][0], head[1] - self.game.snake[-1][1]) if len(self.game.snake) > 1 else (0,0)
        state['snake_length_normalized'] = len(self.game.snake) / (grid_cols * grid_rows)
        state['is_on_edge'] = 1.0 if (head[0]==0 or head[0]==grid_cols-1 or head[1]==0 or head[1]==grid_rows-1) else 0.0
        return state

################################################################################
# SECTION 4: THE ADVANCED HRL AGENT with N-STEP
################################################################################

class HRL_KB_Agent:
    def __init__(self, game_adapter: SnakeGameAdapter):
        self.game = game_adapter; self.kb = SymbolicKnowledgeBase(); self.n_step = 3
        self.options = self._setup_options(); self.meta_controller = self._setup_meta_controller()
        self.active_option: Union[Dict, None] = None; self.option_start_state_vec = None; self.option_cumulative_extrinsic_reward = 0.0
        self.target_update_counter = 0; self.target_update_freq = 100
    def _setup_options(self):
        options = {}
        for name in ["GoToApple", "AvoidTrap"]:
            options[name] = {'name': name, 'online_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), f"{name}_Online"), 'target_network': build_dqn_network(STATE_SIZE, ACTION_SIZE, (64,32), f"{name}_Target"), 'replay_buffer': PrioritizedReplayBuffer(2000), 'batch_size': 32, 'n_step_buffer': deque(maxlen=self.n_step)}
            options[name]['target_network'].set_weights(options[name]['online_network'].get_weights())
        return options
    def _setup_meta_controller(self):
        mc = {"option_names": list(self.options.keys()), "option_indices": {name: i for i, name in enumerate(self.options.keys())}, "online_network": build_dqn_network(STATE_SIZE, len(self.options), (128,64), "Meta_Online"), "target_network": build_dqn_network(STATE_SIZE, len(self.options), (128,64), "Meta_Target"), "replay_buffer": PrioritizedReplayBuffer(5000), "batch_size": 16, "gamma_meta": 0.99, 'n_step_buffer': deque(maxlen=self.n_step)}
        mc['target_network'].set_weights(mc['online_network'].get_weights())
        return mc
    def _update_target_networks(self):
        logger.debug("Updating target networks.")
        for option in self.options.values(): option['target_network'].set_weights(option['online_network'].get_weights())
        self.meta_controller['target_network'].set_weights(self.meta_controller['online_network'].get_weights())
    def _vectorize_state(self, symbolic_state: dict) -> np.ndarray:
        return np.array([symbolic_state.get(key, 0.0) for key in SYMBOLIC_FEATURE_NAMES], dtype=np.float32)
    def get_action(self, symbolic_state: dict) -> int:
        state_vec = self._vectorize_state(symbolic_state)
        if self.active_option:
            opt_name = self.active_option['name']
            terminated = (opt_name == "GoToApple" and (not symbolic_state['is_apple_close'] or symbolic_state['is_trap_close'])) or (opt_name == "AvoidTrap" and not symbolic_state['is_trap_close'])
            if terminated:
                exp = (self.option_start_state_vec, self.meta_controller['option_indices'][opt_name], self.option_cumulative_extrinsic_reward, state_vec, self.game.is_done())
                self._process_n_step(self.meta_controller, exp, self.meta_controller['gamma_meta'])
                self.active_option = None
        if self.active_option is None:
            q_values = self.meta_controller['online_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0]
            for advice in self.kb.reason(symbolic_state):
                if advice['type'] == 'option_bias' and (opt_idx := self.meta_controller['option_indices'].get(advice['option_name'])) is not None: q_values[opt_idx] += advice['bias_value']
            idx = np.argmax(q_values)
            self.active_option = self.options[self.meta_controller['option_names'][idx]]
            self.option_start_state_vec = state_vec; self.option_cumulative_extrinsic_reward = 0.0
        return np.argmax(self.active_option['online_network'].predict(np.expand_dims(state_vec, axis=0), verbose=0)[0])
    def learn_components(self, prev_sym_state, action, reward, next_sym_state, done, global_step):
        if self.active_option:
            opt = self.active_option
            intrinsic_reward=20.0 if opt['name']=="GoToApple" and reward>5 else (prev_sym_state['apple_dist']-next_sym_state['apple_dist'])*5.0 if opt['name']=="GoToApple" else ((next_sym_state['trap_dist']-prev_sym_state['trap_dist'])*2.0 if (next_sym_state['trap_dist']-prev_sym_state['trap_dist'])>0 else -5.0)
            exp = (self._vectorize_state(prev_sym_state), action, intrinsic_reward, self._vectorize_state(next_sym_state), done)
            self._process_n_step(opt, exp, 0.95)
            self.option_cumulative_extrinsic_reward += reward
            if global_step % 4 == 0 and len(opt['replay_buffer']) >= opt['batch_size']: self._train_model(opt, 0.95)
        if done:
            for opt in self.options.values():
                while len(opt['n_step_buffer']) > 0: self._process_n_step(opt, None, 0.95, force_flush=True)
            mc = self.meta_controller
            while len(mc['n_step_buffer']) > 0: self._process_n_step(mc, None, mc['gamma_meta'], force_flush=True)
            if len(mc['replay_buffer']) >= mc['batch_size']: self._train_model(mc, mc['gamma_meta'])
        if self.target_update_counter > 0 and self.target_update_counter % self.target_update_freq == 0: self._update_target_networks()
    def _process_n_step(self, model_dict, new_experience, gamma, force_flush=False):
        if new_experience: model_dict['n_step_buffer'].append(new_experience)
        if len(model_dict['n_step_buffer']) >= self.n_step or (force_flush and len(model_dict['n_step_buffer']) > 0):
            buffer = model_dict['n_step_buffer']
            R = sum([(gamma**i) * buffer[i][2] for i in range(len(buffer))])
            s, a, _, _, _ = buffer[0]
            final_s_prime, final_done = buffer[-1][3], buffer[-1][4]
            model_dict['replay_buffer'].add((s, a, R, final_s_prime, final_done))
            if not force_flush: buffer.popleft()
            else: buffer.clear() # Clear after processing to avoid re-processing
    def _train_model(self, model_dict, gamma):
        self.target_update_counter += 1
        experiences, indices, is_weights = model_dict['replay_buffer'].sample(model_dict['batch_size'])
        s, a, r, ns, d = map(np.array, zip(*experiences))
        online_net = model_dict['online_network']; target_net = model_dict['target_network']
        actions_from_online_net = np.argmax(online_net.predict(ns, verbose=0), axis=1)
        q_values_from_target_net = target_net.predict(ns, verbose=0)
        batch_indices = np.arange(model_dict['batch_size'])
        next_q_values = q_values_from_target_net[batch_indices, actions_from_online_net]
        targets = r + (gamma ** self.n_step) * next_q_values * (1 - d)
        target_f = online_net.predict(s, verbose=0); a = a.astype(int)
        td_errors = targets - target_f[batch_indices, a]
        model_dict['replay_buffer'].update_priorities(indices, td_errors)
        target_f[batch_indices, a] = targets
        online_net.train_on_batch(s, target_f, sample_weight=is_weights)
    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True); self.meta_controller['online_network'].save(os.path.join(path, "meta_controller_q_net.keras"))
        for opt in self.options.values(): opt['online_network'].save(os.path.join(path, f"option_{opt['name']}_policy.keras"))
        logger.info("Models saved successfully.")
    def load_models(self, path="models"):
        try:
            custom_obj = {'NoisyDense': NoisyDense}
            self.meta_controller['online_network'] = models.load_model(os.path.join(path, "meta_controller_q_net.keras"), custom_objects=custom_obj)
            for opt_name in self.options.keys(): self.options[opt_name]['online_network'] = models.load_model(os.path.join(path, f"option_{opt_name}_policy.keras"), custom_objects=custom_obj)
            self._update_target_networks(); logger.info("Models loaded successfully.")
        except Exception as e: logger.warning(f"Could not load saved models from '{path}'. This is expected if architecture changed. Starting from scratch. Error: {e}")

################################################################################
# SECTION 5: THE MAIN RUNNER
################################################################################

def main(args):
    real_game_env = SnakeGame(render=args.render)
    env_adapter = SnakeGameAdapter(real_game_env)
    agent = HRL_KB_Agent(env_adapter)
    agent.load_models()
    
    total_scores, total_steps = [], 0
    for episode in range(1, args.episodes + 1):
        symbolic_state = env_adapter.reset()
        agent.active_option = None; episode_reward = 0; done = False
        
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
            
            if args.render: env_adapter.render()
            if not args.render and total_steps % 100 == 0:
                print(f"\rEpisode: {episode}, Total Steps: {total_steps}", end="")

        if not args.render: print()
        logger.info(f"--- Episode {episode} END --- Score: {real_game_env.score}, Steps: {real_game_env.steps}, Total Reward: {episode_reward:.2f}")
        total_scores.append(real_game_env.score)
        
        if episode % args.save_every == 0 and episode > 0: agent.save_models()
        
    env_adapter.close()
    if total_scores: logger.info(f"\n--- Training Complete ---\nAverage score over {len(total_scores)} episodes: {np.mean(total_scores):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the final integrated HRL agent on the advanced SnakeGame.")
    parser.add_argument("-r","--render",action="store_true",help="Enable graphical rendering.")
    parser.add_argument("-e","--episodes",type=int,default=10000,help="Total episodes to train.")
    parser.add_argument("-s","--save_every",type=int,default=100,help="Save models every N episodes.")
    cli_args = parser.parse_args()
    main(cli_args)
