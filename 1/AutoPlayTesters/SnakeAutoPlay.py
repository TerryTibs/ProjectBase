import pygame
import random
import numpy as np
from abstract_game import AbstractGame  # Import the abstract game class
from tensorflow.keras.models import load_model

# ---------------- Constants ---------------- #
GRID_WIDTH = 800
GRID_HEIGHT = 600
GRID_SIZE = 10
GRID_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
APPLE_COLOR = (255, 0, 0)
SCORE_COLOR = (255, 255, 255)
FONT_SIZE = 14
FONT_NAME = 'Arial'
DEFAULT_FPS = 60  # High FPS for smooth rendering
INITIAL_SNAKE_LENGTH = 3
GRID_COLS = GRID_WIDTH // GRID_SIZE
GRID_ROWS = GRID_HEIGHT // GRID_SIZE

# ---------------- Snake Game Class ---------------- #
class SnakeGame(AbstractGame):
    def __init__(self):
        pygame.init()
        try:
            self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
            pygame.display.set_caption("Snake Game - AI Learning")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

            self.state_size = 6
            self.action_size = 4

            self.fps = DEFAULT_FPS
            self.snake_speed = 5  # Snake moves once every 10 frames
            self.move_counter = 0  # Counter to track movement intervals

            self.reset()
            self.draw()
        except Exception as e:
            print(f"Error during Pygame initialization: {e}")
            raise

    def reset(self):
        self.snake = [(GRID_COLS // 2, GRID_ROWS // 2)]
        for i in range(1, INITIAL_SNAKE_LENGTH):
            self.snake.append((self.snake[0][0] - i, self.snake[0][1]))

        self.direction = (1, 0)
        self.apples = []
        self.generate_apples()
        self.score = 0
        self.apples_eaten = 0
        self.game_over = False
        self.fibonacci_rewards = [1, 1]
        self.move_counter = 0

    def generate_apples(self):
        self.apples = []
        while len(self.apples) < 30:
            apple_x = random.randint(0, GRID_COLS - 1)
            apple_y = random.randint(0, GRID_ROWS - 1)
            apple_position = (apple_x, apple_y)

            if apple_position not in self.snake and apple_position not in self.apples:
                self.apples.append(apple_position)

    def generate_single_apple(self):
        while True:
            apple_x = random.randint(0, GRID_COLS - 1)
            apple_y = random.randint(0, GRID_ROWS - 1)
            apple_position = (apple_x, apple_y)

            if apple_position not in self.snake and apple_position not in self.apples:
                return apple_position

    def get_state(self):
        head_x, head_y = self.snake[0]
        apple_x, apple_y = self.apples[0]

        distance_to_left = head_x / GRID_COLS
        distance_to_right = (GRID_COLS - 1 - head_x) / GRID_COLS
        distance_to_up = head_y / GRID_ROWS
        distance_to_down = (GRID_ROWS - 1 - head_y) / GRID_ROWS

        rel_x = (apple_x - head_x) / GRID_COLS
        rel_y = (apple_y - head_y) / GRID_ROWS

        return np.array([rel_x, rel_y, distance_to_left, distance_to_right, distance_to_up, distance_to_down])

    def step(self, action):
        self.clock.tick(self.fps)
        self.move_counter += 1

        if self.move_counter < self.snake_speed:
            return -0.01, self.game_over

        self.move_counter = 0

        if action == 0:
            new_direction = (0, -1)
        elif action == 1:
            new_direction = (0, 1)
        elif action == 2:
            new_direction = (-1, 0)
        elif action == 3:
            new_direction = (1, 0)
        else:
            new_direction = self.direction

        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        if new_head[0] < 0 or new_head[0] >= GRID_COLS or new_head[1] < 0 or new_head[1] >= GRID_ROWS:
            self.game_over = True
            return -10, self.game_over

        if new_head in self.snake[1:]:
            self.game_over = True
            return -10, self.game_over

        self.snake.insert(0, new_head)

        if new_head in self.apples:
            self.apples.remove(new_head)
            self.apples.append(self.generate_single_apple())
            self.apples_eaten += 1
            self.score += 1
            reward = self.fibonacci_rewards[-1] + self.fibonacci_rewards[-2]
            self.fibonacci_rewards.append(reward)
            return reward, self.game_over

        self.snake.pop()
        return -0.01, self.game_over

    def get_action_space(self):
        return self.action_size

    def is_done(self):
        return self.game_over

    def draw(self):
        self.screen.fill(GRID_COLOR)

        for x, y in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        for x, y in self.apples:
            pygame.draw.rect(self.screen, APPLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def quit(self):
        pygame.quit()

# ---------------- DQN Agent ---------------- #
class DQN_PER_Agent:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def get_action(self, state):
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

# ---------------- Load Model & Run Game ---------------- #
MODEL_PATH = "model.keras"
game = SnakeGame()
agent = DQN_PER_Agent(MODEL_PATH)

def autoplay():
    running = True
    while running:
        state = game.get_state()
        action = agent.get_action(state)
        _, game_over = game.step(action)
        game.draw()

        if game_over:
            game.reset()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                game.quit()
                pygame.quit()
                return

if __name__ == "__main__":
    autoplay()

