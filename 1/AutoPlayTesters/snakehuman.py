# Import required libraries
import pygame # For game creation
import random # For random number generation

# ---------------- Constants ---------------- #
GRID_WIDTH = 800 # Define grid width
GRID_HEIGHT = 600 # Define grid height
GRID_SIZE = 10 # Define grid size
GRID_COLOR = (0, 0, 0) # Black color
SNAKE_COLOR = (0, 255, 0) # Green color
APPLE_COLOR = (255, 0, 0) # Red color
SCORE_COLOR = (255, 255, 255) # White color
FONT_SIZE = 20 # Font size for score
FONT_NAME = 'Arial' # Font name for score
DEFAULT_FPS = 60  # Default frames per second
INITIAL_SNAKE_LENGTH = 1 # Initial snake length
GRID_COLS = GRID_WIDTH // GRID_SIZE # Calculate number of columns
GRID_ROWS = GRID_HEIGHT // GRID_SIZE # Calculate number of rows
NUM_APPLES = 100  # Number of apples to spawn

# ---------------- Snake Game Class ---------------- #
class SnakeGame:
    """
    Implements the Snake game.
    """
    def __init__(self):
        """
        Initializes the Snake game, setting up Pygame and game variables.
        """
        pygame.init() # Initialize pygame
        try:
            self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT)) # Create the game window
            pygame.display.set_caption("Snake Game") # Set the window title
            self.clock = pygame.time.Clock() # Create a clock object for controlling FPS
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE) # Set the font

            # Independent speed settings
            self.fps = DEFAULT_FPS  # Frames per second
            self.snake_speed = 5  # Snake moves once every 'snake_speed' frames
            self.move_counter = 0  # Counter to track movement intervals

            self.reset() # Reset the game state
        except Exception as e:
            print(f"Error during Pygame initialization: {e}") # Print the error message
            raise # Raise the exception

    def reset(self):
        """
        Resets the game state, placing the snake and apples in their initial positions.
        """
        self.snake = [(GRID_COLS // 2, GRID_ROWS // 2)] # Initialize the snake at the center
        #Add more segments to the snake, the head starts in the center
        for i in range(1, INITIAL_SNAKE_LENGTH):
            self.snake.append((self.snake[0][0] - i, self.snake[0][1])) # Add segments to the snake

        self.direction = (1, 0) # Set the initial direction (right)
        self.apples = []  # Initialize the apples list
        self.generate_apples() # Generate initial set of apples
        self.score = 0 # Reset the score
        self.game_over = False # Reset game over flag
        self.move_counter = 0  # Reset movement counter

    def generate_apples(self):
        """
        Generates a list of apple positions to have NUM_APPLES apples on the grid.
        """
        self.apples = [] #clear previous apples
        while len(self.apples) < NUM_APPLES: # Ensure there are NUM_APPLES apples
            apple_x = random.randint(0, GRID_COLS - 1) # Generate random x coordinate
            apple_y = random.randint(0, GRID_ROWS - 1) # Generate random y coordinate
            apple_position = (apple_x, apple_y) # Create apple position tuple

            if apple_position not in self.snake and apple_position not in self.apples: # Make sure apple is not on snake
                self.apples.append(apple_position) # Add the position to the list

    def generate_single_apple(self):
        """
        Generates a single apple at a random location, avoiding the snake.
        """
        while True:
            apple_x = random.randint(0, GRID_COLS - 1) # Generate random x coordinate
            apple_y = random.randint(0, GRID_ROWS - 1) # Generate random y coordinate
            apple_position = (apple_x, apple_y) # Create apple position tuple

            if apple_position not in self.snake and apple_position not in self.apples: # Make sure apple is not on snake
                return apple_position # Return the apple position

    def move(self):
        """
        Moves the snake based on the current direction.
        Snake only moves if enough frames have passed, controlling speed.
        """

        self.move_counter += 1  # Increment movement counter

        # Move the snake only if enough frames have passed
        if self.move_counter < self.snake_speed:
            return  # Skip movement this frame

        self.move_counter = 0  # Reset counter

        head_x, head_y = self.snake[0] # Get current head coordinates
        new_head = (head_x + self.direction[0], head_y + self.direction[1]) # Calculate new head coordinates

        # Check for game over conditions (wall collision or self-collision)
        if new_head[0] < 0 or new_head[0] >= GRID_COLS or new_head[1] < 0 or new_head[1] >= GRID_ROWS:
            self.game_over = True # Game over if snake hits the wall
            return

        if new_head in self.snake[1:]:
            self.game_over = True # Game over if snake hits itself
            return

        self.snake.insert(0, new_head) # Insert the new head into the snake

        if new_head in self.apples:  # Check if the new head is on an apple
            self.apples.remove(new_head) # Remove the apple
            self.apples.append(self.generate_single_apple()) # Add a new apple
            self.score += 1 # Increment score
        else:
            self.snake.pop() # Remove the last segment of the snake

    def draw(self):
        """
        Renders the game state visually using Pygame.
        """
        self.screen.fill(GRID_COLOR) # Fill screen with grid color

        for x, y in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)) # Draw the snake

        for x, y in self.apples:
            pygame.draw.rect(self.screen, APPLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)) # Draw the apples

        score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR) # Render the score
        self.screen.blit(score_text, (10, 10)) # Blit the score to the screen

        pygame.display.flip() # Update the full display Surface to the screen

    def run(self):
        """
        Runs the main game loop.
        """
        running = True
        while running:
            self.clock.tick(self.fps)  # Control the FPS

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.direction != (0, 1):
                        self.direction = (0, -1)
                    elif event.key == pygame.K_DOWN and self.direction != (0, -1):
                        self.direction = (0, 1)
                    elif event.key == pygame.K_LEFT and self.direction != (1, 0):
                        self.direction = (-1, 0)
                    elif event.key == pygame.K_RIGHT and self.direction != (-1, 0):
                        self.direction = (1, 0)

            if not self.game_over:
                self.move() # Move the snake
                self.draw() # Draw the game state
            else:
                # Display Game Over message
                game_over_text = self.font.render("Game Over! Press SPACE to restart.", True, SCORE_COLOR)
                text_rect = game_over_text.get_rect(center=(GRID_WIDTH // 2, GRID_HEIGHT // 2))
                self.screen.blit(game_over_text, text_rect)
                pygame.display.flip()

                # Wait for spacebar to restart or ESC to quit
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                self.reset()
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                running = False
                                waiting = False

        pygame.quit() # Quit pygame

# ---------------- Main Execution ---------------- #
if __name__ == '__main__':
    game = SnakeGame() # Create a SnakeGame object
    game.run() # Run the game
