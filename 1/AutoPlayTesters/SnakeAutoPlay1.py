import pygame # Handles game rendering and events - PYGAME LIBRARY
import random # Used for random number generation - RANDOM NUMBERS
import numpy as np # Used for numerical operations - NUMPY LIBRARY
from abstract_game import AbstractGame  # Import the abstract game class - IMPORT ABSTRACT BASE CLASS
from tensorflow.keras.models import load_model # Import load_model - LOAD TRAINED MODELS

# ---------------- Constants ---------------- #
# Constants - GAME CONSTANTS
GRID_WIDTH = 800 # Grid width - GRID WIDTH
GRID_HEIGHT = 600 # Grid height - GRID HEIGHT
GRID_SIZE = 10 # Grid size - GRID SIZE
GRID_COLOR = (0, 0, 0) # Grid color - GRID COLOR
SNAKE_COLOR = (0, 255, 0) # Snake color - SNAKE COLOR
APPLE_COLOR = (255, 0, 0) # Apple color - APPLE COLOR
SCORE_COLOR = (255, 255, 255) # Score color - SCORE COLOR
FONT_SIZE = 14 # Font size for score - FONT SIZE
FONT_NAME = 'Arial' # Font name for score - FONT NAME
DEFAULT_FPS = 60  # High FPS for smooth rendering - FRAMES PER SECOND
INITIAL_SNAKE_LENGTH = 3 # Initial snake length - INITIAL SNAKE LENGTH
GRID_COLS = GRID_WIDTH // GRID_SIZE # Calculate number of grid columns - GRID COLUMNS
GRID_ROWS = GRID_HEIGHT // GRID_SIZE # Calculate number of grid rows - GRID ROWS

# ---------------- Snake Game Class ---------------- #
class SnakeGame(AbstractGame):
    """Snake game environment based on the AbstractGame class."""
    def __init__(self):
        """Initialize the Snake game environment."""
        pygame.init() # Initialize pygame - INITIALIZE PYGAME
        try: # TRY TO INITIALIZE THE GAME
            self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT)) # Create the game window - CREATE THE SCREEN
            pygame.display.set_caption("Snake Game - AI Learning") # Set the window title - WINDOW TITLE
            self.clock = pygame.time.Clock() # Create a clock object - GAME CLOCK
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE) # Create a font object - GAME FONT

            self.state_size = 6 # Define the state size - STATE SIZE
            self.action_size = 4 # Define the action size - ACTION SIZE

            self.fps = DEFAULT_FPS # Set the frames per second - FRAMES PER SECOND
            self.snake_speed = 5  # Snake moves once every 10 frames - SNAKE SPEED
            self.move_counter = 0  # Counter to track movement intervals - MOVE COUNTER

            self.reset() # Reset the game - RESET THE GAME
            self.draw() # Draw the game - DRAW THE GAME
        except Exception as e: # HANDLE EXCEPTIONS
            print(f"Error during Pygame initialization: {e}") # Print error message - PRINT MESSAGE
            raise # Re-raise exception - RE-RAISE

    def reset(self):
        """Reset the game state."""
        self.snake = [(GRID_COLS // 2, GRID_ROWS // 2)] # Initialize snake at center of the grid - INITIALIZE SNAKE
        for i in range(1, INITIAL_SNAKE_LENGTH): # Add additional segments to the snake - ADD SEGMENTS
            self.snake.append((self.snake[0][0] - i, self.snake[0][1])) # Append the segment - ADD SEGMENT

        self.direction = (1, 0) # Set the initial direction - SET DIRECTION
        self.apples = [] # Initialize apples list - INITIALIZE APPLES
        self.generate_apples() # Generate apples - GENERATE APPLES
        self.score = 0 # Set initial score - SET SCORE
        self.apples_eaten = 0 # Set initial apples eaten - SET APPLES EATEN
        self.game_over = False # Set game over to False - GAME OVER
        self.fibonacci_rewards = [10, 10, 30, 50, 80] # Initialize fibonacci rewards - INITIALIZE REWARDS
        self.move_counter = 0 # Reset the move counter - RESET MOVE COUNTER

    def generate_apples(self):
        """Generate a list of apple positions."""
        self.apples = [] # Clear the apples list - CLEAR APPLES
        while len(self.apples) < 30: # Generate 30 apples - GENERATE 30 APPLES
            apple_x = random.randint(0, GRID_COLS - 1) # Generate a random x coordinate - RANDOM X
            apple_y = random.randint(0, GRID_ROWS - 1) # Generate a random y coordinate - RANDOM Y
            apple_position = (apple_x, apple_y) # Create apple position tuple - APPLE POSITION

            if apple_position not in self.snake and apple_position not in self.apples: # Check for invalid positions - CHECK FOR INVALID POSITIONS
                self.apples.append(apple_position) # Append the apple position to the apples list - ADD APPLE

    def generate_single_apple(self):
        """Generate a single apple at a valid position."""
        while True: # Loop until a valid position is found - INFINITE LOOP
            apple_x = random.randint(0, GRID_COLS - 1) # Generate a random x coordinate - RANDOM X
            apple_y = random.randint(0, GRID_ROWS - 1) # Generate a random y coordinate - RANDOM Y
            apple_position = (apple_x, apple_y) # Create apple position tuple - APPLE POSITION

            if apple_position not in self.snake and apple_position not in self.apples: # Check for invalid positions - CHECK FOR INVALID POSITIONS
                return apple_position # Return the valid apple position - RETURN APPLE

    def get_state(self):
        """Get the current state of the game."""
        head_x, head_y = self.snake[0] # Get the x, y coordinates of the snake's head - HEAD COORDINATES
        apple_x, apple_y = self.apples[0] # Get the x, y coordinates of the first apple - APPLE COORDINATES

        distance_to_left = head_x / GRID_COLS # Calculate the distance to the left wall - LEFT DISTANCE
        distance_to_right = (GRID_COLS - 1 - head_x) / GRID_COLS # Calculate the distance to the right wall - RIGHT DISTANCE
        distance_to_up = head_y / GRID_ROWS # Calculate the distance to the top wall - UP DISTANCE
        distance_to_down = (GRID_ROWS - 1 - head_y) / GRID_ROWS # Calculate the distance to the bottom wall - DOWN DISTANCE

        rel_x = (apple_x - head_x) / GRID_COLS # Calculate the relative x coordinate of the apple - RELATIVE X
        rel_y = (apple_y - head_y) / GRID_ROWS # Calculate the relative y coordinate of the apple - RELATIVE Y

        return np.array([rel_x, rel_y, distance_to_left, distance_to_right, distance_to_up, distance_to_down]) # Return the state array - RETURN STATE

    def step(self, action):
        """Take a game step given the action."""
        self.clock.tick(self.fps) # Limit the frame rate - LIMIT FRAME RATE
        self.move_counter += 1 # Increment the move counter - INCREMENT COUNTER

        if self.move_counter < self.snake_speed: # Check if enough frames have passed - CHECK FRAMES
            return -0.01, self.game_over # Return a small penalty - PENALTY

        self.move_counter = 0 # Reset the move counter - RESET COUNTER

        if action == 0: # Move up - MOVE UP
            new_direction = (0, -1) # Set the new direction to up - SET DIRECTION
        elif action == 1: # Move down - MOVE DOWN
            new_direction = (0, 1) # Set the new direction to down - SET DIRECTION
        elif action == 2: # Move left - MOVE LEFT
            new_direction = (-1, 0) # Set the new direction to left - SET DIRECTION
        elif action == 3: # Move right - MOVE RIGHT
            new_direction = (1, 0) # Set the new direction to right - SET DIRECTION
        else: # Invalid action - INVALID ACTION
            new_direction = self.direction # Keep the same direction - KEEP DIRECTION

        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction: # Check if the new direction is the opposite of the current direction - CHECK DIRECTION
            self.direction = new_direction # Set the direction - SET DIRECTION

        head_x, head_y = self.snake[0] # Get the coordinates of the head - HEAD COORDINATES
        new_head = (head_x + self.direction[0], head_y + self.direction[1]) # Calculate the new head position - NEW HEAD POSITION

        if new_head[0] < 0 or new_head[0] >= GRID_COLS or new_head[1] < 0 or new_head[1] >= GRID_ROWS: # Check for collision with the walls - COLLISION WITH WALL
            self.game_over = True # Game over - GAME OVER
            return -1000, self.game_over # Return a large penalty - PENALTY

        if new_head in self.snake[1:]: # Check for collision with the snake's body - COLLISION WITH BODY
            self.game_over = True # Game over - GAME OVER
            return -1000, self.game_over # Return a large penalty - PENALTY

        self.snake.insert(0, new_head) # Insert the new head into the snake list - INSERT HEAD

        if new_head in self.apples: # Check if the snake ate an apple - APPLE EATEN
            self.apples.remove(new_head) # Remove the apple - REMOVE APPLE
            self.apples.append(self.generate_single_apple()) # Add a new apple - ADD APPLE
            self.apples_eaten += 1 # Increment apples eaten - INCREMENT APPLES EATEN
            self.score += 1 # Increment the score - INCREMENT SCORE
            reward = self.fibonacci_rewards[-1] + self.fibonacci_rewards[-2] # Calculate the reward - CALCULATE REWARD
            self.fibonacci_rewards.append(reward) # Append reward - APPEND REWARD
            return reward, self.game_over # Return the reward - RETURN REWARD

        self.snake.pop() # Remove the last element from the snake list - REMOVE TAIL
        return -0.01, self.game_over # Return a small penalty - PENALTY

    def get_action_space(self):
        """Get the number of possible actions."""
        return self.action_size # Return the action space size - RETURN ACTION SPACE

    def is_done(self):
        """Check if the game is over."""
        return self.game_over # Return the game over status - RETURN GAME OVER

    def draw(self):
        """Draw the game state to the screen."""
        self.screen.fill(GRID_COLOR) # Fill the screen with the grid color - DRAW GRID

        for x, y in self.snake: # Iterate through the snake's body - ITERATE SNAKE
            pygame.draw.rect(self.screen, SNAKE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)) # Draw the snake - DRAW SNAKE

        for x, y in self.apples: # Iterate through the apples - ITERATE APPLES
            pygame.draw.rect(self.screen, APPLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)) # Draw the apples - DRAW APPLES

        score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR) # Render the score text - RENDER SCORE
        self.screen.blit(score_text, (10, 10)) # Draw the score text on the screen - DRAW SCORE

        pygame.display.flip() # Update the display - UPDATE DISPLAY

    def quit(self):
        """Quit the game."""
        pygame.quit() # Quit pygame - QUIT PYGAME

# ---------------- DQN Agent Class ---------------- #
class DQN_PER_Agent:
    """DQN agent with PER."""
    def __init__(self, model_path):
        """Initialize the DQN agent."""
        self.model = load_model(model_path) # Load the model - LOAD MODEL
    
    def get_action(self, state):
        """Get an action from the model."""
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0) # Predict Q-values - PREDICT Q-VALUES
        return np.argmax(q_values[0]) # Return the action with the highest Q-value - RETURN ACTION

# ---------------- Load Model & Run Game ---------------- #
# Load Model & Run Game - LOAD MODEL AND RUN
MODEL_PATH = "model.keras" # Model file path - MODEL PATH
game = SnakeGame() # Create the Snake Game - CREATE THE GAME
agent = DQN_PER_Agent(MODEL_PATH) # Create the DQN agent - CREATE THE AGENT

def autoplay():
    """Run the game with the DQN agent."""
    running = True # Running flag - RUNNING
    while running: # Game loop - GAME LOOP
        state = game.get_state() # Get the game state - GET STATE
        action = agent.get_action(state) # Get the action from the agent - GET ACTION
        _, game_over = game.step(action) # Take a step in the game - TAKE STEP
        game.draw() # Draw the game - DRAW GAME

        if game_over: # Check if the game is over - GAME OVER CHECK
            game.reset() # Reset the game - RESET GAME

        for event in pygame.event.get(): # Check for events - CHECK EVENTS
            if event.type == pygame.QUIT: # Check for quit event - QUIT EVENT
                running = False # Set running to False - STOP RUNNING
                game.quit() # Quit the game - QUIT GAME
                pygame.quit() # Quit pygame - QUIT PYGAME
                return # Return - RETURN

if __name__ == "__main__":
    autoplay() # Run the autoplay function - RUN AUTOPLAY
