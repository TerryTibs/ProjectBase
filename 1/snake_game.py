# --- START OF FILE project/snake_game.py ---

# Import required libraries
import pygame
import random
import numpy as np
from abstract_game import AbstractGame
from collections import deque
import logging # Use logging for warnings/errors

# ---------------- Constants ---------------- #
GRID_WIDTH = 800
GRID_HEIGHT = 600
GRID_SIZE = 10 # Size of each grid cell in pixels for drawing
GRID_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
APPLE_COLOR = (255, 0, 0)
SCORE_COLOR = (255, 255, 255)
FONT_SIZE = 14
FONT_NAME = 'Arial' # Default Pygame font used if not found
DEFAULT_FPS = 60 # Target FPS for rendering and clock ticks
INITIAL_SNAKE_LENGTH = 1
GRID_COLS = GRID_WIDTH // GRID_SIZE
GRID_ROWS = GRID_HEIGHT // GRID_SIZE

# State/Action Sizes
ACTION_SIZE = 4 # 0:Up, 1:Down, 2:Left, 3:Right
STATE_SIZE = 28 # Size for 1D Vector State representation

# Game Mechanics & Rewards
SNAKE_SPEED = 3 # Number of environment steps per snake move (higher = slower)
BASE_LIMIT = 500000 # Base time limit in frames (steps) before episode ends
NUM_APPLES = 100 # Number of apples on screen simultaneously
DEATH_PENALTY = -500 # Penalty for dying (collision)
APPLE_REWARD = 10 # Reward for eating an apple
WALL_PROXIMITY_PENALTY_SCALE = 0.01 # Scaler for wall proximity penalty
APPLE_PROXIMITY_REWARD_SCALE = 0.01 # Scaler for apple proximity reward
MOVE_PENALTY = -0.001 # Small penalty for each move step (if not eating)
TIMEOUT_PENALTY = -1 # Penalty for exceeding frame limit (adaptive)
FOOD_DENSITY_RADIUS = 25 # Radius in grid cells for food density calculation

# Constants for 2D Grid State Representation (Normalized float values)
GRID_STATE_EMPTY = 0.0
GRID_STATE_WALL = 0.0 # Can treat walls same as empty for snake movement logic
GRID_STATE_BODY = 0.25
GRID_STATE_HEAD = 0.75 # Make head distinct from body
GRID_STATE_APPLE = 1.0


# --------------------- SnakeGame Class --------------------- #
class SnakeGame(AbstractGame):
    """
    Implements the Snake game, inheriting from AbstractGame.
    Provides three state representations:
    1. 1D feature vector (`get_state()`)
    2. 2D grid map (`get_grid_state()`)
    3. 3D pixel screen (`get_screen()`)
    """

    def __init__(self, render=True):
        """
        Initializes the Snake game, setting up Pygame and game variables.
        Args:
            render (bool): Whether to initialize Pygame display and render graphics.
        """
        super().__init__() # Initialize base class attributes (state_size, action_size defined below)
        pygame.init()
        self.render = render

        # Define state dimensions/sizes
        self.state_size = STATE_SIZE # For 1D Vector agents (from get_state)
        self.action_size = ACTION_SIZE
        self._screen_size = (GRID_HEIGHT, GRID_WIDTH, 3) # H, W, C for pixel state
        self.grid_state_shape = (GRID_ROWS, GRID_COLS) # R, C for grid state

        # Setup Pygame display and font if rendering is enabled
        self.screen = None
        self.font = None
        if self.render:
            try:
                self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
                pygame.display.set_caption("Snake Game - AI Learning")
            except pygame.error as e:
                 logging.error(f"Failed to set Pygame display mode: {e}", exc_info=True)
                 self.render = False # Disable rendering if display fails
            if self.render: # Check again in case display failed
                try:
                    self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
                except pygame.error:
                    logging.warning(f"Font '{FONT_NAME}' not found. Using default Pygame font.")
                    self.font = pygame.font.Font(None, FONT_SIZE + 4) # Use default font if specific one fails

        self.clock = pygame.time.Clock()
        self.fps = DEFAULT_FPS

        # Game mechanics variables
        self.snake_speed = max(1, SNAKE_SPEED) # Ensure speed is at least 1
        self.move_counter = 0

        # Internal game state variables (initialized in reset)
        self.snake = []
        self.apples = []
        self.direction = (0, 0)
        self.score = 0
        self.apples_eaten = 0
        self.game_over = False
        self.frame_iteration = 0
        self.last_actions = deque(maxlen=3)
        self.velocity_history = deque(maxlen=3)
        self.will_grow = False
        self.frame_last_ate = 0

        # Initialize game state by calling reset
        self.reset()

    # --- Core AbstractGame Methods ---

    def reset(self):
        """Resets the game to its initial state."""
        # Center snake initially
        start_x = GRID_COLS // 2
        start_y = GRID_ROWS // 2
        self.snake = [(start_x, start_y)]
        # Extend initial snake to the left if length > 1
        for i in range(1, INITIAL_SNAKE_LENGTH):
            x_pos = start_x - i
            if x_pos >= 0: # Ensure initial body is within bounds
                 self.snake.append((x_pos, start_y))

        self.direction = (1, 0) # Start moving right
        self.apples = []
        self.generate_apples() # Generate initial apple(s)
        self.score = 0
        self.apples_eaten = 0
        self.game_over = False
        self.move_counter = 0
        self.frame_iteration = 0 # Total frames elapsed in this episode
        self.last_actions = deque(maxlen=3)
        self.velocity_history = deque(maxlen=3)
        self.velocity_history.append(self.snake[0]) # Initial position
        self.will_grow = False
        self.frame_last_ate = 0

        # Return initial state? The abstract class notes say the training loop
        # calls get_state()/get_screen() after reset, so returning None is fine.
        return None

    def step(self, action):
        """
        Takes an action, updates the game state, calculates reward, and checks if done.
        Args:
            action (int): The action chosen by the agent (0=Up, 1=Down, 2=Left, 3=Right).
        Returns:
            tuple: (reward, done) where reward is float, done is bool.
        """
        self.frame_iteration += 1
        reward = 0.0 # Initialize reward for this step

        # Handle Pygame events and clock tick only if rendering
        if self.render:
            self.clock.tick(self.fps)
            # Process Pygame events (like window close)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                    logging.info("Pygame quit event received during step.")
                    # Return a specific penalty for quitting, maybe less than death
                    return DEATH_PENALTY / 2, self.game_over
            # Note: Drawing is now handled by the training loop if needed, not here.

        # Control snake movement speed
        self.move_counter += 1
        if self.move_counter < self.snake_speed:
            # If not time to move, return 0 reward and current 'done' status
            return 0.0, self.game_over

        self.move_counter = 0 # Reset move counter

        # --- Action Handling ---
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # 0:Up, 1:Down, 2:Left, 3:Right
        if not (0 <= action < len(directions)):
            logging.warning(f"Invalid action received: {action}. Keeping current direction.")
            new_direction = self.direction # Keep current direction if action is invalid
        else:
            new_direction = directions[action]

        # Prevent snake reversing direction
        current_dx, current_dy = self.direction
        new_dx, new_dy = new_direction
        # Check only if snake has more than one segment
        if len(self.snake) > 1 and new_dx == -current_dx and new_dy == -current_dy:
            # If reversal attempt, keep the current direction instead
            new_direction = self.direction
        else:
            # Update direction if not a reversal or snake is length 1
            self.direction = new_direction

        # --- Calculate New Head Position ---
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        # --- Check Collisions ---
        # 1. Wall Collision
        if new_head[0] < 0 or new_head[0] >= GRID_COLS or new_head[1] < 0 or new_head[1] >= GRID_ROWS:
            self.game_over = True
            reward = DEATH_PENALTY
            return reward, self.game_over

        # 2. Self Collision
        # Need to check against the snake body *before* potentially removing the tail
        if new_head in self.snake:
            self.game_over = True
            reward = DEATH_PENALTY
            return reward, self.game_over

        # --- Move Snake ---
        # If no collision, insert the new head
        self.snake.insert(0, new_head)

        # --- Check Apple Collision & Handle Growth ---
        ate_apple = False
        if new_head in self.apples:
            self.apples.remove(new_head)
            self.generate_apples() # Replenish apples
            self.apples_eaten += 1
            self.score += 1
            reward = APPLE_REWARD # Base reward for eating apple
            ate_apple = True
            self.will_grow = True # Flag that tail shouldn't be popped this step
            self.frame_last_ate = self.frame_iteration
        else:
            # If no apple eaten, remove the tail segment
            self.snake.pop()
            self.will_grow = False
            # Apply movement penalty only if no apple was eaten
            reward = MOVE_PENALTY

        # --- Proximity Rewards/Penalties (Reward Shaping) ---
        # Wall proximity penalty (inversely proportional to distance)
        wall_distance_px = self._distance_to_walls(return_grid=False)
        reward -= WALL_PROXIMITY_PENALTY_SCALE / max(1.0, wall_distance_px) # Avoid div by zero

        # Apple proximity reward (inversely proportional to distance)
        apple_distance_px = self._distance_to_closest_apple(return_grid=False)
        if apple_distance_px != float('inf'):
             reward += APPLE_PROXIMITY_REWARD_SCALE / max(1.0, apple_distance_px) # Avoid div by zero

        # --- Adaptive Time Limit Check ---
        adaptive_limit = BASE_LIMIT + 5 * len(self.snake) # Limit increases with snake length
        if self.frame_iteration > adaptive_limit:
            self.game_over = True
            reward += TIMEOUT_PENALTY # Add penalty for timeout
            logging.debug(f"Game ended due to timeout ({self.frame_iteration} > {adaptive_limit})")

        # --- Update History Trackers ---
        self.velocity_history.append(self.snake[0]) # Store new head position
        self.last_actions.append(action) # Store action taken

        # Return final reward and done status for this step
        return reward, self.game_over

    def get_state(self) -> np.ndarray:
        """
        Returns the current game state as a 1D NumPy array (feature vector).
        Features are based on grid coordinates and normalized or are flags.
        State size should match self.state_size (e.g., 28).
        """
        head_x, head_y = self.snake[0]

        # Features 1-4: Normalized distance to walls
        distance_to_left = head_x / (GRID_COLS - 1) if GRID_COLS > 1 else 0.0
        distance_to_right = (GRID_COLS - 1 - head_x) / (GRID_COLS - 1) if GRID_COLS > 1 else 0.0
        distance_to_up = head_y / (GRID_ROWS - 1) if GRID_ROWS > 1 else 0.0
        distance_to_down = (GRID_ROWS - 1 - head_y) / (GRID_ROWS - 1) if GRID_ROWS > 1 else 0.0

        # Features 5-6: Normalized relative coordinates to the first apple
        # If no apples, use head's own position (relative coords = 0)
        apple_x, apple_y = self.apples[0] if self.apples else (head_x, head_y)
        rel_x = (apple_x - head_x) / (GRID_COLS - 1) if GRID_COLS > 1 else 0.0
        rel_y = (apple_y - head_y) / (GRID_ROWS - 1) if GRID_ROWS > 1 else 0.0

        # Features 7-10: Immediate danger (wall or self) check in absolute directions
        danger_abs_up = 1.0 if self._check_danger(head_x, head_y - 1) else 0.0
        danger_abs_down = 1.0 if self._check_danger(head_x, head_y + 1) else 0.0
        danger_abs_left = 1.0 if self._check_danger(head_x - 1, head_y) else 0.0
        danger_abs_right = 1.0 if self._check_danger(head_x + 1, head_y) else 0.0

        # Feature 11: Normalized minimum grid distance to any wall
        min_grid_dist_to_wall = self._distance_to_walls(return_grid=True)
        # Max possible min distance is to center of grid
        max_min_grid_distance = min(GRID_COLS // 2, GRID_ROWS // 2)
        wall_distance_normalized = min_grid_dist_to_wall / max_min_grid_distance if max_min_grid_distance > 0 else 0.0

        # Feature 12: Normalized minimum grid distance to closest apple
        min_apple_distance_grid = self._distance_to_closest_apple(return_grid=True)
        # Max possible distance is diagonal of grid
        max_grid_distance_diag = np.sqrt((GRID_COLS - 1)**2 + (GRID_ROWS - 1)**2) if GRID_COLS > 1 and GRID_ROWS > 1 else 1.0
        apple_distance_normalized = min_apple_distance_grid / max_grid_distance_diag if min_apple_distance_grid != float('inf') and max_grid_distance_diag > 0 else 1.0 # 1.0 if no apple

        # Features 13-14: Snake Body Direction (relative direction from neck to head)
        body_direction_x, body_direction_y = (0, 0)
        if len(self.snake) > 1:
             # Sign gives -1 (left/up), 0 (no change), 1 (right/down)
             body_direction_x = np.sign(self.snake[0][0] - self.snake[1][0])
             body_direction_y = np.sign(self.snake[0][1] - self.snake[1][1])

        # Features 15-18: Nearest Food Direction (One-Hot Encoded)
        nearest_food_direction_idx = self._get_nearest_food_direction(head_x, head_y) # 0:Up, 1:Down, 2:Left, 3:Right
        food_dir_up = 1.0 if nearest_food_direction_idx == 0 else 0.0
        food_dir_down = 1.0 if nearest_food_direction_idx == 1 else 0.0
        food_dir_left = 1.0 if nearest_food_direction_idx == 2 else 0.0
        food_dir_right = 1.0 if nearest_food_direction_idx == 3 else 0.0

        # Feature 19: Food Density (normalized count of apples within radius)
        food_density = self._calculate_food_density(head_x, head_y)

        # Feature 20: Snake Length (normalized by grid area)
        max_possible_length = GRID_COLS * GRID_ROWS
        snake_length_norm = len(self.snake) / max_possible_length if max_possible_length > 0 else 0.0

        # Features 21-22: Velocity (change in grid coords over last step)
        velocity_x, velocity_y = self._calculate_velocity() # Returns -1, 0, or 1

        # Feature 23: Time Since Last Apple (normalized by adaptive limit)
        time_since_last_apple = self.frame_iteration - self.frame_last_ate
        adaptive_limit = BASE_LIMIT + 5 * len(self.snake) # Recalculate adaptive limit
        time_since_apple_norm = min(1.0, time_since_last_apple / max(1.0, adaptive_limit)) # Avoid div by zero

        # Feature 24: Snake Body Proximity (Normalized inverse distance)
        min_body_distance_grid = self._distance_to_closest_body()
        # Closer distance -> higher value (1 = very close/touching, 0 = far)
        snake_body_proximity_norm = 1.0 - (min_body_distance_grid / max_grid_distance_diag) if min_body_distance_grid != float('inf') and max_grid_distance_diag > 0 else 0.0
        snake_body_proximity_norm = max(0.0, min(1.0, snake_body_proximity_norm)) # Clamp to [0, 1]

        # Features 25-28: Current Moving Direction (One-Hot Encoded)
        is_up = 1.0 if self.direction == (0, -1) else 0.0
        is_down = 1.0 if self.direction == (0, 1) else 0.0
        is_left = 1.0 if self.direction == (-1, 0) else 0.0
        is_right = 1.0 if self.direction == (1, 0) else 0.0

        # Assemble the state vector
        state = [
            distance_to_left, distance_to_right, distance_to_up, distance_to_down, # 1-4
            rel_x, rel_y, # 5-6
            danger_abs_up, danger_abs_down, danger_abs_left, danger_abs_right, # 7-10
            wall_distance_normalized,       # 11
            apple_distance_normalized,      # 12
            body_direction_x, body_direction_y, # 13-14
            food_dir_up, food_dir_down, food_dir_left, food_dir_right, # 15-18
            food_density,                   # 19
            snake_length_norm,              # 20
            velocity_x, velocity_y,         # 21-22
            time_since_apple_norm,          # 23
            snake_body_proximity_norm,      # 24
            is_up, is_down, is_left, is_right # 25-28
        ]

        # Final check for correct length
        if len(state) != self.state_size:
             logging.error(f"State vector construction error: Expected size {self.state_size}, got {len(state)}")
             # Pad or truncate to ensure correct size, though this indicates a bug
             state = (state + [0.0] * self.state_size)[:self.state_size]

        return np.array(state, dtype=np.float32)

    def get_grid_state(self) -> np.ndarray:
        """
        Returns the current game state as a 2D grid representation (Rows x Cols).
        Values are floats representing: Empty, Body, Head, Apple.
        """
        # Start with an empty grid
        grid = np.full(self.grid_state_shape, GRID_STATE_EMPTY, dtype=np.float32)

        # Place apples onto the grid
        for x, y in self.apples:
            # Check bounds before placing
            if 0 <= y < GRID_ROWS and 0 <= x < GRID_COLS:
                grid[y, x] = GRID_STATE_APPLE

        # Place snake body segments (excluding the head)
        # Iterate backwards through the list excluding the head (index 0)
        for i in range(len(self.snake) - 1, 0, -1):
            x, y = self.snake[i]
            if 0 <= y < GRID_ROWS and 0 <= x < GRID_COLS:
                grid[y, x] = GRID_STATE_BODY

        # Place the snake head last (overwrites body/apple if necessary)
        head_x, head_y = self.snake[0]
        if 0 <= head_y < GRID_ROWS and 0 <= head_x < GRID_COLS:
            grid[head_y, head_x] = GRID_STATE_HEAD # Use grid[y, x] indexing

        return grid # Shape (Rows, Cols)

    def get_screen(self) -> np.ndarray:
        """
        Captures the 3D game screen (pixels) and returns it as a NumPy array (H, W, C).
        Returns zeros if rendering is disabled or Pygame display is unavailable.
        """
        if not self.render or self.screen is None:
            # Return a blank screen if rendering is disabled or display failed
            return np.zeros(self._screen_size, dtype=np.uint8) # HWC

        # Get the currently displayed Pygame surface
        screen_surface = pygame.display.get_surface()
        if screen_surface is None:
             logging.warning("Attempted to get screen, but Pygame surface is None.")
             return np.zeros(self._screen_size, dtype=np.uint8)

        # Convert Pygame surface to NumPy array (usually Width x Height x Channels)
        screen_array = pygame.surfarray.array3d(screen_surface)

        # Transpose to standard Height x Width x Channels format if needed
        # Pygame often returns WxHxC, most DL frameworks expect HxWxC
        if screen_array.shape[0] == GRID_WIDTH and screen_array.shape[1] == GRID_HEIGHT:
             return np.transpose(screen_array, (1, 0, 2))
        elif screen_array.shape[0] == GRID_HEIGHT and screen_array.shape[1] == GRID_WIDTH:
             return screen_array # Already in HxWxC format (less common)
        else:
             logging.error(f"Unexpected screen array shape from Pygame: {screen_array.shape}. Expected H={GRID_HEIGHT}, W={GRID_WIDTH}.")
             return np.zeros(self._screen_size, dtype=np.uint8) # Return blank on error

    def is_done(self) -> bool:
        """Checks whether the game episode has ended."""
        return self.game_over

    def get_action_space(self) -> int:
        """Returns the number of possible discrete actions."""
        return self.action_size

    def draw(self):
        """
        Renders the current game state graphically using Pygame.
        Only does something if rendering is enabled and Pygame display is initialized.
        """
        if not self.render or self.screen is None or self.font is None:
            return # Cannot draw if rendering disabled or Pygame setup failed

        # Fill background
        self.screen.fill(GRID_COLOR)

        # Draw snake segments
        for x, y in self.snake:
            # Ensure coordinates are valid before drawing
            if 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS:
                 pygame.draw.rect(self.screen, SNAKE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw apples
        for x, y in self.apples:
             if 0 <= x < GRID_COLS and 0 <= y < GRID_ROWS:
                 pygame.draw.rect(self.screen, APPLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Draw score text
        score_text_surface = self.font.render(f"Score: {self.score}", True, SCORE_COLOR)
        self.screen.blit(score_text_surface, (10, 10))

        # Update the display to show drawn elements
        pygame.display.flip()

    def quit(self):
        """Shuts down the Pygame environment properly."""
        logging.info("Quitting Pygame.")
        pygame.quit()

    # --- Size/Shape Information Methods ---

    def get_state_size(self) -> int:
        """Returns the size of the 1D feature vector state."""
        return self.state_size

    def get_grid_shape(self) -> tuple:
        """Returns the shape of the 2D grid state (Rows, Cols)."""
        return self.grid_state_shape

    def get_screen_size(self) -> tuple:
        """Returns the 3D pixel screen size (H, W, C)."""
        return self._screen_size

    # --- Helper Methods for Game Logic & State Calculation ---

    def generate_apples(self):
        """Generates apple positions until NUM_APPLES are on the grid, avoiding collisions."""
        attempt_limit = GRID_COLS * GRID_ROWS * 2 # Generous limit
        attempts = 0
        spawned_count = 0
        # Ensure self.apples is initialized as a list
        if not isinstance(self.apples, list): self.apples = []

        # Keep adding apples until the desired number is reached or limit exceeded
        while len(self.apples) < NUM_APPLES and attempts < attempt_limit:
            apple_position = self.generate_single_apple()
            if apple_position: # If a valid position was found
                self.apples.append(apple_position)
                spawned_count += 1
            attempts += 1

        # Log a warning if not all apples could be placed
        if len(self.apples) < NUM_APPLES:
             logging.warning(f"Could not place all {NUM_APPLES} apples after {attempt_limit} attempts. Total apples: {len(self.apples)}.")

    def generate_single_apple(self):
        """Generates a single apple at a random location, avoiding snake and other apples."""
        # Try multiple times to find an empty spot
        attempt_limit = GRID_COLS * GRID_ROWS
        for _ in range(attempt_limit):
            apple_pos = (random.randint(0, GRID_COLS - 1), random.randint(0, GRID_ROWS - 1))
            # Check collision with snake body and other existing apples
            if apple_pos not in self.snake and apple_pos not in self.apples:
                return apple_pos # Found a valid spot

        # Return None if no valid spot found after many attempts
        logging.debug("Failed to find a unique spot for a new apple.")
        return None

    def _check_danger(self, x, y):
        """Checks if a coordinate (x, y) is outside bounds or occupied by the snake's body (not head)."""
        # Check wall collision
        if x < 0 or x >= GRID_COLS or y < 0 or y >= GRID_ROWS:
            return True
        # Check self-collision (only against body segments, index 1 onwards)
        if (x, y) in self.snake[1:]:
            return True
        return False

    def _distance_to_walls(self, return_grid=True):
        """Calculates min distance from head to any wall (in grid cells or pixels)."""
        head_x, head_y = self.snake[0]
        # Distances to left, right, top, bottom walls
        dist_grid = min(head_x, GRID_COLS - 1 - head_x, head_y, GRID_ROWS - 1 - head_y)
        # Ensure non-negative distance
        dist_grid = max(0, dist_grid)
        return dist_grid if return_grid else dist_grid * GRID_SIZE

    def _distance_to_closest_apple(self, return_grid=True):
        """Calculates Euclidean distance from head to the closest apple (grid or pixels)."""
        if not self.apples:
            return float('inf') # No apples exist
        head_pos = np.array(self.snake[0])
        apple_positions = np.array(self.apples)
        # Calculate squared Euclidean distances, then take sqrt of minimum
        distances_sq = np.sum((apple_positions - head_pos)**2, axis=1)
        min_dist_sq = np.min(distances_sq)
        min_dist_grid = np.sqrt(min_dist_sq)
        return min_dist_grid if return_grid else min_dist_grid * GRID_SIZE

    def _get_nearest_food_direction(self, head_x, head_y):
        """Determines the index (0-3) representing the general direction of the nearest food."""
        if not self.apples: return 0 # Default to Up if no apples

        # Find the apple with the minimum squared distance
        nearest_apple_pos = min(self.apples, key=lambda pos: (head_x - pos[0])**2 + (head_y - pos[1])**2)

        # Calculate vector from head to nearest apple
        dx = nearest_apple_pos[0] - head_x
        dy = nearest_apple_pos[1] - head_y

        # Determine primary direction based on largest absolute difference
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2 # Right (dx+) or Left (dx-)
        else:
            # If abs(dy) >= abs(dx)
            return 1 if dy > 0 else 0 # Down (dy+) or Up (dy-)
            # This correctly handles the case where dx=0, dy=0 (on apple) -> Up (0)

    def _calculate_food_density(self, head_x, head_y):
        """Calculates normalized food density (apples within radius / total apples)."""
        if NUM_APPLES <= 0 or FOOD_DENSITY_RADIUS <= 0: return 0.0

        radius_sq = FOOD_DENSITY_RADIUS**2
        # Count apples within the squared radius
        count = sum(1 for ax, ay in self.apples if (head_x - ax)**2 + (head_y - ay)**2 <= radius_sq)

        # Normalize by the total number of apples currently configured
        return count / NUM_APPLES

    def _calculate_velocity(self):
        """Calculates velocity vector (dx, dy) based on head movement since last step."""
        if len(self.velocity_history) < 2:
            # Not enough history to calculate velocity (e.g., first step)
            return 0, 0

        # Get current and previous head positions
        x1, y1 = self.velocity_history[-1] # Current
        x0, y0 = self.velocity_history[-2] # Previous

        # Return sign of change to get unit vector (-1, 0, or 1)
        return np.sign(x1 - x0), np.sign(y1 - y0)

    def _distance_to_closest_body(self):
        """Calculates minimum Euclidean distance from head to any body segment."""
        if len(self.snake) <= 1:
            # No body segments to collide with
            return float('inf')

        head_pos = np.array(self.snake[0])
        # Body segments start from index 1
        body_positions = np.array(self.snake[1:])

        # Handle case where body_positions might be empty if snake length just became 1? Unlikely.
        if body_positions.size == 0: return float('inf')

        # Calculate squared Euclidean distances, find minimum, take sqrt
        distances_sq = np.sum((body_positions - head_pos)**2, axis=1)
        min_dist_sq = np.min(distances_sq)

        # Avoid issues with floating point precision if distance is extremely close to zero
        min_dist = np.sqrt(min_dist_sq)
        return min_dist if min_dist > 1e-6 else float('inf') # Treat near-zero as infinity (collision likely)

# --- END OF FILE project/snake_game.py ---
