import pygame
import numpy as np

# Constants for the debug window
DEBUG_WIN_WIDTH = 300
DEBUG_WIN_HEIGHT = 400
DEBUG_GRID_COLS = 15  # Conceptual grid size around the snake head
DEBUG_GRID_ROWS = 15
DEBUG_CELL_SIZE = 20 # Fixed pixel size for debug grid cells
DEBUG_GRID_WIDTH_PX = DEBUG_GRID_COLS * DEBUG_CELL_SIZE
DEBUG_GRID_HEIGHT_PX = DEBUG_GRID_ROWS * DEBUG_CELL_SIZE
DEBUG_BG_COLOR = (50, 50, 50)
DEBUG_GRID_COLOR = (100, 100, 100)
DEBUG_SNAKE_HEAD_COLOR = (0, 255, 0) # Green
DEBUG_APPLE_COLOR = (255, 0, 0) # Red
DEBUG_DANGER_COLOR = (255, 100, 0) # Orange
DEBUG_TEXT_COLOR = (200, 200, 200)
DEBUG_FONT_SIZE = 16
DEBUG_FONT_NAME = 'Arial'

class DebugVisualizer:
    def __init__(self):
        """Initializes the debug visualization window."""
        print("Initializing Debug Visualizer...")
        self.is_initialized = False
        try:
            self.screen = pygame.display.set_mode((DEBUG_WIN_WIDTH, DEBUG_WIN_HEIGHT))
            pygame.display.set_caption("Agent State Visualization")
            self.font = pygame.font.SysFont(DEBUG_FONT_NAME, DEBUG_FONT_SIZE)
            self.clock = pygame.time.Clock() # Separate clock if needed
            self.head_pos_grid = (DEBUG_GRID_COLS // 2, DEBUG_GRID_ROWS // 2) # Center
            self.latest_state = None
            self.is_initialized = True
            print("Debug Visualizer Initialized.")
        except Exception as e:
            print(f"Error initializing Debug Visualizer Pygame window: {e}")
            self.screen = None # Ensure screen is None if init fails

    def update(self, state_vector):
        """Receives the latest state vector from the agent."""
        if not self.is_initialized: return
        # Ensure state_vector is a numpy array and has the expected size (25)
        if isinstance(state_vector, np.ndarray) and state_vector.shape == (25,):
             self.latest_state = state_vector
        else:
             # Print a warning if state is not as expected, but don't store it
             print(f"Warning: DebugVisualizer received invalid state format. Shape: {getattr(state_vector, 'shape', 'N/A')}")
             self.latest_state = None # Clear latest state if invalid


    def draw(self):
        """Draws the visualization based on the latest state."""
        if not self.is_initialized or self.screen is None: return

        # Handle events for this window (important to keep it responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Technically closing this window shouldn't stop training,
                # but makes sense to signal closure if possible.
                # For simplicity, just let it close visually.
                print("Debug window closed by user.")
                self.is_initialized = False # Stop drawing updates
                pygame.display.quit() # Quit this specific display context if possible? No, use flag.
                self.screen = None # Mark as closed
                return # Stop drawing

        self.screen.fill(DEBUG_BG_COLOR)

        # --- Draw Grid ---
        grid_origin_x = (DEBUG_WIN_WIDTH - DEBUG_GRID_WIDTH_PX) // 2
        grid_origin_y = 10 # Padding from top
        for row in range(DEBUG_GRID_ROWS + 1):
            y = grid_origin_y + row * DEBUG_CELL_SIZE
            pygame.draw.line(self.screen, DEBUG_GRID_COLOR, (grid_origin_x, y), (grid_origin_x + DEBUG_GRID_WIDTH_PX, y))
        for col in range(DEBUG_GRID_COLS + 1):
            x = grid_origin_x + col * DEBUG_CELL_SIZE
            pygame.draw.line(self.screen, DEBUG_GRID_COLOR, (x, grid_origin_y), (x, grid_origin_y + DEBUG_GRID_HEIGHT_PX))

        # --- Draw Snake Head ---
        head_px_x = grid_origin_x + self.head_pos_grid[0] * DEBUG_CELL_SIZE
        head_px_y = grid_origin_y + self.head_pos_grid[1] * DEBUG_CELL_SIZE
        pygame.draw.rect(self.screen, DEBUG_SNAKE_HEAD_COLOR, (head_px_x, head_px_y, DEBUG_CELL_SIZE, DEBUG_CELL_SIZE))

        if self.latest_state is None:
            # Display message if state is not available
            text_surf = self.font.render("Waiting for state...", True, DEBUG_TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(DEBUG_WIN_WIDTH // 2, DEBUG_WIN_HEIGHT - 30))
            self.screen.blit(text_surf, text_rect)
            pygame.display.flip() # Update display
            return

        # --- Visualize State Features ---
        try:
            # Indices based on the snake.py get_state() feature list
            # 0-3: Wall distances (normalized 0-1, 0=close)
            dist_l, dist_r, dist_u, dist_d = self.latest_state[0:4]
            # 4-5: Relative apple position (normalized -1 to 1 approx)
            rel_x, rel_y = self.latest_state[4:6]
            # 6-9: Immediate Danger flags (0/1)
            danger_u, danger_d, danger_l, danger_r = self.latest_state[6:10]
            # 10: Normalized wall dist (grid)
            wall_dist_norm = self.latest_state[10]
            # 11: Normalized apple dist (grid)
            apple_dist_norm = self.latest_state[11]
            # 12-13: Body direction X, Y (-1,0,1)
            body_dir_x, body_dir_y = self.latest_state[12:14]
            # 14: Nearest food direction index (0-3)
            nearest_food_idx = int(self.latest_state[14])
            # 15: Food density (0-1)
            food_density = self.latest_state[15]
            # 16: Snake length norm (0-1)
            snake_len_norm = self.latest_state[16]
            # 17-18: Velocity X, Y (-1,0,1)
            vel_x, vel_y = self.latest_state[17:19]
            # 19: Time until next move (0-1)
            time_next_move = self.latest_state[19]
            # 20: Body proximity norm (grid)
            body_prox_norm = self.latest_state[20]
            # 21-24: Direction one-hot (0/1)
            is_up, is_down, is_left, is_right = self.latest_state[21:25]

            # --- Draw Danger Zones ---
            if danger_u > 0.5: # Use > 0.5 for robustness with float comparison
                pygame.draw.rect(self.screen, DEBUG_DANGER_COLOR, (head_px_x, head_px_y - DEBUG_CELL_SIZE, DEBUG_CELL_SIZE, DEBUG_CELL_SIZE))
            if danger_d > 0.5:
                pygame.draw.rect(self.screen, DEBUG_DANGER_COLOR, (head_px_x, head_px_y + DEBUG_CELL_SIZE, DEBUG_CELL_SIZE, DEBUG_CELL_SIZE))
            if danger_l > 0.5:
                pygame.draw.rect(self.screen, DEBUG_DANGER_COLOR, (head_px_x - DEBUG_CELL_SIZE, head_px_y, DEBUG_CELL_SIZE, DEBUG_CELL_SIZE))
            if danger_r > 0.5:
                pygame.draw.rect(self.screen, DEBUG_DANGER_COLOR, (head_px_x + DEBUG_CELL_SIZE, head_px_y, DEBUG_CELL_SIZE, DEBUG_CELL_SIZE))

            # --- Draw Relative Apple Position ---
            # rel_x/y are normalized grid diff. Scale by half the debug grid size.
            # E.g., rel_x = 0.1 means slightly right. rel_x = 1.0 means far right.
            # Map rel_x range (approx -1 to 1) to cell offset range (-cols/2 to +cols/2)
            apple_offset_x = rel_x * (DEBUG_GRID_COLS / 2.0)
            apple_offset_y = rel_y * (DEBUG_GRID_ROWS / 2.0)

            apple_grid_x = int(self.head_pos_grid[0] + apple_offset_x)
            apple_grid_y = int(self.head_pos_grid[1] + apple_offset_y)

            # Clamp apple position within debug grid bounds
            apple_grid_x = max(0, min(DEBUG_GRID_COLS - 1, apple_grid_x))
            apple_grid_y = max(0, min(DEBUG_GRID_ROWS - 1, apple_grid_y))

            apple_px_x = grid_origin_x + apple_grid_x * DEBUG_CELL_SIZE
            apple_px_y = grid_origin_y + apple_grid_y * DEBUG_CELL_SIZE
            pygame.draw.rect(self.screen, DEBUG_APPLE_COLOR, (apple_px_x, apple_px_y, DEBUG_CELL_SIZE, DEBUG_CELL_SIZE))

            # --- Draw Text Readouts ---
            text_y_start = grid_origin_y + DEBUG_GRID_HEIGHT_PX + 10
            line_height = self.font.get_linesize()

            # Line 1: Wall & Apple Distances
            text1 = f"WallDist:{wall_dist_norm:.2f} AppleDist:{apple_dist_norm:.2f} BodyProx:{body_prox_norm:.2f}"
            surf1 = self.font.render(text1, True, DEBUG_TEXT_COLOR)
            self.screen.blit(surf1, (10, text_y_start))

            # Line 2: Length, Velocity, Food Dir/Density
            dir_map = {0: "U", 1: "D", 2: "L", 3: "R", -1: "?"} # Add default
            food_dir_char = dir_map.get(nearest_food_idx, "?")
            text2 = f"Len:{snake_len_norm:.2f} Vel:({int(vel_x)},{int(vel_y)}) FoodDir:{food_dir_char} Dens:{food_density:.2f}"
            surf2 = self.font.render(text2, True, DEBUG_TEXT_COLOR)
            self.screen.blit(surf2, (10, text_y_start + line_height))

            # Line 3: Current Direction & Next Move Time
            current_dir_str = ""
            if is_up > 0.5: current_dir_str="Up"
            elif is_down > 0.5: current_dir_str="Down"
            elif is_left > 0.5: current_dir_str="Left"
            elif is_right > 0.5: current_dir_str="Right"
            text3 = f"Dir: {current_dir_str} NxtMove:{time_next_move:.2f}"
            surf3 = self.font.render(text3, True, DEBUG_TEXT_COLOR)
            self.screen.blit(surf3, (10, text_y_start + 2 * line_height))

        except IndexError:
            # Handle case where state vector might be shorter than expected
            text_surf = self.font.render("State vector error!", True, (255, 0, 0))
            text_rect = text_surf.get_rect(center=(DEBUG_WIN_WIDTH // 2, DEBUG_WIN_HEIGHT - 30))
            self.screen.blit(text_surf, text_rect)
        except Exception as e:
            # Catch other potential drawing errors
            print(f"Error during debug draw: {e}")
            text_surf = self.font.render("Draw Error!", True, (255, 0, 0))
            text_rect = text_surf.get_rect(center=(DEBUG_WIN_WIDTH // 2, DEBUG_WIN_HEIGHT - 30))
            self.screen.blit(text_surf, text_rect)

        pygame.display.flip() # Update this window
        self.clock.tick(30) # Limit update rate of this window

    def close(self):
        """Closes the debug window."""
        if self.is_initialized and self.screen:
            print("Closing Debug Visualizer window.")
            pygame.display.quit() # Should only affect this window's display context
            self.is_initialized = False
            self.screen = None

# --- END OF FILE debug_visualizer.py ---
