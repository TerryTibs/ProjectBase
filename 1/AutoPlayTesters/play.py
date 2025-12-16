import pygame
import random
import numpy as np
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
DEFAULT_FPS = 60
INITIAL_SNAKE_LENGTH = 1
GRID_COLS = GRID_WIDTH // GRID_SIZE
GRID_ROWS = GRID_HEIGHT // GRID_SIZE

# ---------------- Snake Game Class ---------------- #
class SnakeGame:
    """Snake game environment"""
    def __init__(self, model):
        """Initialize the Snake game environment with a pre-trained model."""
        pygame.init()
        self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
        pygame.display.set_caption("Snake Game - AI Playing")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

        self.model = model  # Load the model
        self.state_size = 25  # State size for the model
        self.action_size = 4  # Action size for the model

        self.reset()

    def reset(self):
        """Reset the game state."""
        self.snake = [(GRID_COLS // 2, GRID_ROWS // 2)]
        for i in range(1, INITIAL_SNAKE_LENGTH):
            self.snake.append((self.snake[0][0] - i, self.snake[0][1]))

        self.direction = (1, 0)
        self.apples = []
        self.generate_apples()
        self.score = 0
        self.game_over = False

    def generate_apples(self):
        """Generate a list of apple positions."""
        self.apples = []
        while len(self.apples) < 100:
            apple_x = random.randint(0, GRID_COLS - 1)
            apple_y = random.randint(0, GRID_ROWS - 1)
            apple_position = (apple_x, apple_y)

            if apple_position not in self.snake and apple_position not in self.apples:
                self.apples.append(apple_position)

    def generate_single_apple(self):
        """Generate a single apple at a valid position."""
        while True:
            apple_x = random.randint(0, GRID_COLS - 1)
            apple_y = random.randint(0, GRID_ROWS - 1)
            apple_position = (apple_x, apple_y)

            if apple_position not in self.snake and apple_position not in self.apples:
                return apple_position

    def get_state(self):
        """Get the current state of the game."""
        head_x, head_y = self.snake[0]
        apple_x, apple_y = self.apples[0]

        # Calculate distances to walls and relative apple position
        distance_to_left = head_x / GRID_COLS
        distance_to_right = (GRID_COLS - 1 - head_x) / GRID_COLS
        distance_to_up = head_y / GRID_ROWS
        distance_to_down = (GRID_ROWS - 1 - head_y) / GRID_ROWS

        rel_x = (apple_x - head_x) / GRID_COLS
        rel_y = (apple_y - head_y) / GRID_ROWS

        # Create the state vector (length 25)
        state = [
            rel_x, rel_y, distance_to_left, distance_to_right, distance_to_up, distance_to_down
        ]

        # Add placeholders to make the state length 25
        for _ in range(19):
            state.append(0)

        return np.array(state).reshape((1, self.state_size))

    def step(self, action):
        """Take a game step given the action."""
        if self.game_over:
            return 0, self.game_over

        head_x, head_y = self.snake[0]
        if action == 0:  # Up
            new_direction = (0, -1)
        elif action == 1:  # Down
            new_direction = (0, 1)
        elif action == 2:  # Left
            new_direction = (-1, 0)
        elif action == 3:  # Right
            new_direction = (1, 0)

        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction

        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        if new_head[0] < 0 or new_head[0] >= GRID_COLS or new_head[1] < 0 or new_head[1] >= GRID_ROWS:
            self.game_over = True
            return -1000, self.game_over  # Large penalty for hitting the wall

        if new_head in self.snake[1:]:
            self.game_over = True
            return -1000, self.game_over  # Large penalty for hitting the snake

        self.snake.insert(0, new_head)

        if new_head in self.apples:
            self.apples.remove(new_head)
            self.apples.append(self.generate_single_apple())
            self.score += 1
            return 100, self.game_over  # Reward for eating an apple

        self.snake.pop()
        return -0.1, self.game_over  # Small penalty for normal movement

    def get_action(self):
        """Get the action from the model."""
        state = self.get_state()
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def draw(self):
        """Draw the game state to the screen."""
        self.screen.fill(GRID_COLOR)
        for x, y in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        for apple_x, apple_y in self.apples:
            pygame.draw.rect(self.screen, APPLE_COLOR, (apple_x * GRID_SIZE, apple_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

# ---------------- Main Loop ---------------- #
def play_game(model):
    """Play the game using the model."""
    game = SnakeGame(model)

    while not game.game_over:
        game.draw()

        # Get action from model
        action = game.get_action()

        # Take a step in the game
        reward, game_over = game.step(action)

        # If game is over, break the loop
        if game_over:
            print(f"Game Over! Final Score: {game.score}")
            break

# Load the pre-trained model
model = load_model('model.keras')

# Start the game
play_game(model)

