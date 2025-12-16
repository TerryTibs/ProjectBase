# standalone_hrl_kb_agent_v4.py
# A Proof of Concept for a Hierarchical Reinforcement Learning Agent
# guided by a Symbolic Knowledge Base, as architected by "Bob".
# VERSION 4: Corrected TypeError in Option class constructors.

import pygame
import random
import numpy as np
from collections import deque, defaultdict
import time
import logging
from typing import Union, Dict, List, Any, Tuple

# --- Basic Logging Configuration ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("HRL_KB_PoC")


# ---------------------------------------------------------------------------- #
#                            PART 1: THE ENVIRONMENT                           #
# ---------------------------------------------------------------------------- #

class AbstractGame:
    """Abstract base class for game environments."""
    def reset(self): raise NotImplementedError
    def step(self, action): raise NotImplementedError
    def get_symbolic_state(self) -> Dict: raise NotImplementedError
    def get_action_space_size(self) -> int: raise NotImplementedError
    def is_done(self) -> bool: raise NotImplementedError
    def render(self): pass
    def close(self): pass


class SimpleSnakeGame(AbstractGame):
    """
    A simplified Snake game that provides a rich symbolic state dictionary.
    """
    def __init__(self, render=True, grid_size=10):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = 30
        self.width = self.height = self.grid_size * self.cell_size
        self.render_mode = render
        self.prev_player_pos = None
        self.apple_pos = None
        self.trap_pos = None
        self.obstacle_pos = None

        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("HRL-KB Agent Proof of Concept")
            self.font = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()
        
        self.action_space_size = 4
        self.action_map = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)} # 0:N, 1:S, 2:W, 3:E

        self.reset()

    def reset(self):
        self.snake = deque([(self.grid_size // 2, self.grid_size // 2)])
        self.direction = (1, 0)
        self.apple_pos = self._place_item()
        self.trap_pos = self._place_item()
        self.obstacle_pos = self._place_item()
        self.score = 0
        self.steps = 0
        self.max_steps = self.grid_size * 5
        self.done = False
        logger.info(f"Env Reset: Player@{self.snake[0]}, Apple@{self.apple_pos}, Trap@{self.trap_pos}, Obstacle@{self.obstacle_pos}")
        return self.get_symbolic_state()

    def _place_item(self):
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            occupied_spaces = list(self.snake)
            if self.apple_pos is not None: occupied_spaces.append(self.apple_pos)
            if self.trap_pos is not None: occupied_spaces.append(self.trap_pos)
            if self.obstacle_pos is not None: occupied_spaces.append(self.obstacle_pos)
            if pos not in occupied_spaces:
                return pos

    def get_symbolic_state(self) -> Dict:
        state = {}
        head = self.snake[0]
        
        state['apple_dx'] = self.apple_pos[0] - head[0]
        state['apple_dy'] = self.apple_pos[1] - head[1]
        state['trap_dx'] = self.trap_pos[0] - head[0]
        state['trap_dy'] = self.trap_pos[1] - head[1]
        state['obstacle_dx'] = self.obstacle_pos[0] - head[0]
        state['obstacle_dy'] = self.obstacle_pos[1] - head[1]
        
        state['apple_dist'] = np.linalg.norm([state['apple_dx'], state['apple_dy']])
        state['trap_dist'] = np.linalg.norm([state['trap_dx'], state['trap_dy']])
        
        state['is_apple_close'] = 1.0 if state['apple_dist'] < (self.grid_size / 4) else 0.0
        state['is_trap_close'] = 1.0 if state['trap_dist'] < 2.5 else 0.0
        
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
        if self.done: return self.get_symbolic_state(), 0, True, {}
        
        self.steps += 1
        reward = -0.1
        self.prev_player_pos = self.snake[0]

        new_dir = self.action_map[action]
        if len(self.snake) > 1 and new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]:
            pass
        else:
            self.direction = new_dir
        
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        if self.steps >= self.max_steps:
            self.done = True
            reward -= 1.0
            logger.info("Max steps reached.")

        if self._is_danger(new_head):
            self.done = True
            reward = -10.0
            if new_head == self.trap_pos: logger.warning("TERMINAL STATE: Stepped on a trap!")
            elif new_head in self.snake: logger.warning("TERMINAL STATE: Collided with self!")
            elif not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size): logger.warning("TERMINAL STATE: Hit a wall!")
            elif new_head == self.obstacle_pos: logger.warning("TERMINAL STATE: Hit an obstacle!")
            return self.get_symbolic_state(), reward, self.done, {}

        self.snake.appendleft(new_head)

        if new_head == self.apple_pos:
            self.score += 1
            reward = 10.0
            self.apple_pos = self._place_item()
            self.trap_pos = self._place_item()
            self.max_steps += self.grid_size * 2
            logger.info("APPLE EATEN! Score: {}".format(self.score))
        else:
            self.snake.pop()

        return self.get_symbolic_state(), reward, self.done, {}

    def get_action_space_size(self) -> int: return self.action_space_size
    def is_done(self) -> bool: return self.done

    def render(self):
        if not self.render_mode: return
        self.screen.fill((20, 20, 20))
        for pos in self.snake: pygame.draw.rect(self.screen, (0,200,0), (pos[0]*self.cell_size, pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (200,0,0), (self.apple_pos[0]*self.cell_size, self.apple_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255,100,0), (self.trap_pos[0]*self.cell_size, self.trap_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (100,100,100), (self.obstacle_pos[0]*self.cell_size, self.obstacle_pos[1]*self.cell_size, self.cell_size, self.cell_size))
        score_text = self.font.render(f"Score: {self.score}", True, (255,255,255))
        self.screen.blit(score_text, (5, 5))
        pygame.display.flip()
        self.clock.tick(5)

    def close(self): pygame.quit()


# ---------------------------------------------------------------------------- #
#                PART 2: CORE SYMBOLIC & HRL BASE CLASSES                      #
# ---------------------------------------------------------------------------- #

class GlobalBlackboard:
    def __init__(self):
        self.data = {}
        self.timestamp = 0
        logger.info("GlobalBlackboard created.")

    def post(self, key: str, value: Any, source: str):
        self.timestamp += 1
        self.data[key] = {'value': value, 'source': source, 'ts': self.timestamp}
        logger.info(f"BB POST by {source}: {key} = {value}")

    def read(self, key: str) -> Union[Any, None]:
        entry = self.data.get(key)
        return entry['value'] if entry else None

    def get_full_snapshot(self) -> Dict:
        return self.data.copy()

class AbstractOption:
    def __init__(self, option_name: str, num_actions: int):
        self.option_name = option_name
        self.num_actions = num_actions
    def get_action(self, symbolic_state: Dict) -> int: raise NotImplementedError
    def is_terminated(self, symbolic_state: Dict) -> bool: raise NotImplementedError

# ---------------------------------------------------------------------------- #
#                  PART 3: AGENT AND ITS REASONING COMPONENTS                  #
# ---------------------------------------------------------------------------- #

class SymbolicKnowledgeBase:
    def __init__(self):
        self.rules = []
        self._add_predefined_rules()
        logger.info("SymbolicKnowledgeBase initialized with predefined rules.")

    def _add_predefined_rules(self):
        self.rules.append((
            "HighPriorityAppleRule",
            lambda s: s.get('is_apple_close') == 1.0 and \
                      (s.get('danger_up') + s.get('danger_down') + s.get('danger_left') + s.get('danger_right')) == 0.0,
            {'type': 'option_bias', 'option_name': 'GoToApple', 'bias_value': 5.0}
        ))
        
        self.rules.append((
            "UrgentTrapAvoidanceRule",
            lambda s: s.get('is_trap_close') == 1.0,
            {'type': 'option_bias', 'option_name': 'AvoidTrap', 'bias_value': 10.0}
        ))

    def reason(self, symbolic_state: Dict) -> List[Dict]:
        advice_list = []
        for name, condition_fn, consequence in self.rules:
            if condition_fn(symbolic_state):
                advice = consequence.copy()
                advice['reason'] = name
                advice_list.append(advice)
                logger.debug(f"KB Rule Fired: '{name}'")
        return advice_list

class GoToAppleOption(AbstractOption):
    # FIX: Added a constructor that calls the parent constructor
    def __init__(self, num_actions: int):
        super().__init__(option_name="GoToApple", num_actions=num_actions)

    def get_action(self, symbolic_state: Dict) -> int:
        dx = symbolic_state['apple_dx']
        dy = symbolic_state['apple_dy']
        if abs(dx) > abs(dy): return 3 if dx > 0 else 2
        else: return 1 if dy > 0 else 0

    def is_terminated(self, symbolic_state: Dict) -> bool:
        return symbolic_state['is_apple_close'] == 0.0 or symbolic_state['is_trap_close'] == 1.0

class AvoidTrapOption(AbstractOption):
    # FIX: Added a constructor that calls the parent constructor
    def __init__(self, num_actions: int):
        super().__init__(option_name="AvoidTrap", num_actions=num_actions)
    
    def get_action(self, symbolic_state: Dict) -> int:
        dx = symbolic_state['trap_dx']
        dy = symbolic_state['trap_dy']
        if abs(dx) > abs(dy): return 2 if dx > 0 else 3
        else: return 0 if dy > 0 else 1
        
    def is_terminated(self, symbolic_state: Dict) -> bool:
        return symbolic_state['is_trap_close'] == 0.0

class MetaController:
    def __init__(self, options: List[AbstractOption], knowledge_base: SymbolicKnowledgeBase):
        self.options = {opt.option_name: opt for opt in options}
        self.kb = knowledge_base
        self.option_base_preferences = {"GoToApple": 1.0, "AvoidTrap": 0.5}
        logger.info("MetaController initialized.")

    def select_option(self, symbolic_state: Dict) -> AbstractOption:
        option_scores = self.option_base_preferences.copy()
        kb_advice = self.kb.reason(symbolic_state)
        
        for advice in kb_advice:
            if advice['type'] == 'option_bias':
                opt_name = advice['option_name']
                if opt_name in option_scores:
                    option_scores[opt_name] += advice['bias_value']
                    logger.info(f"Meta: Applying bias of {advice['bias_value']} to '{opt_name}' due to rule '{advice['reason']}'.")

        best_option_name = max(option_scores, key=option_scores.get)
        logger.info(f"Meta: Option scores: {option_scores}. Selected: '{best_option_name}'")
        return self.options[best_option_name]

class HRL_KB_Agent:
    def __init__(self, game: SimpleSnakeGame, blackboard: GlobalBlackboard):
        self.game = game
        self.blackboard = blackboard
        
        self.kb = SymbolicKnowledgeBase()
        self.options = [GoToAppleOption(game.get_action_space_size()), 
                        AvoidTrapOption(game.get_action_space_size())]
        self.meta_controller = MetaController(self.options, self.kb)
        
        self.active_option: Union[AbstractOption, None] = None
        logger.info("HRL_KB_Agent created and fully initialized.")

    def get_action(self, symbolic_state: Dict) -> int:
        if self.active_option and self.active_option.is_terminated(symbolic_state):
            logger.info(f"Option '{self.active_option.option_name}' terminated.")
            self.active_option = None
        
        if self.active_option is None:
            self.active_option = self.meta_controller.select_option(symbolic_state)
            self.blackboard.post("active_option_name", self.active_option.option_name, "HRL_Agent_Meta")
        
        return self.active_option.get_action(symbolic_state)


# ---------------------------------------------------------------------------- #
#                             PART 4: MAIN RUNNER                              #
# ---------------------------------------------------------------------------- #
def main():
    logger.info("--- Initializing Standalone HRL-KB Proof of Concept ---")
    
    env = SimpleSnakeGame(render=True)
    blackboard = GlobalBlackboard()
    agent = HRL_KB_Agent(env, blackboard)
    
    num_episodes = 100
    total_scores = []

    for episode in range(num_episodes):
        logger.info(f"\n{'='*20} Episode {episode + 1}/{num_episodes} {'='*20}")
        symbolic_state = env.reset()
        agent.active_option = None
        episode_reward = 0
        done = False
        
        while not done:
            if env.render_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        logger.warning("Quit command received. Shutting down.")
                        env.close()
                        return
            
            logger.info(f"\n-- Step {env.steps + 1} --")
            state_str = ", ".join([f"{k}: {v:.2f}" for k,v in symbolic_state.items()])
            logger.info(f"Current Symbolic State: {{ {state_str} }}")
            
            action = agent.get_action(symbolic_state)
            
            # Improved logging to prevent error if active_option is None at this point (should not happen, but safe)
            active_option_name = agent.active_option.option_name if agent.active_option else "None"
            logger.info(f"Agent Action: {action} (Active Option: {active_option_name})")

            next_symbolic_state, reward, done, _ = env.step(action)
            
            logger.info(f"Outcome: Reward={reward:.2f}, Done={done}")

            episode_reward += reward
            symbolic_state = next_symbolic_state

            if env.render_mode:
                env.render()
                time.sleep(0.3)
        
        logger.info(f"--- Episode {episode + 1} END --- Final Score: {env.score}, Total Reward: {episode_reward:.2f}\n")
        total_scores.append(env.score)
        if episode < num_episodes - 1:
            time.sleep(1.5)

    env.close()
    logger.info(f"\n--- Proof of Concept Complete ---")
    logger.info(f"Average score over {num_episodes} episodes: {np.mean(total_scores):.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("An error occurred during execution.", exc_info=True)
        if 'pygame' in locals() and pygame.get_init():
             pygame.quit()
