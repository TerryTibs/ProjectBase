import pygame # Handles game rendering and events - PYGAME LIBRARY
import random # Used for random number generation - RANDOM NUMBERS

# Initialize pygame - INITIALIZE PYGAME
pygame.init() # INITIALIZE PYGAME

# ---------------- Constants ---------------- #
# Constants - GAME CONSTANTS
WIDTH, HEIGHT = 800, 600 # Screen dimensions - SCREEN DIMENSIONS
TILE_SIZE = 20 # Size of each tile in the maze - TILE SIZE
BALL_RADIUS = TILE_SIZE // 2 # Radius of the ball - BALL RADIUS
WHITE = (255, 255, 255) # White color - WHITE
BLACK = (0, 0, 0) # Black color - BLACK
YELLOW = (255, 255, 0) # Yellow color - YELLOW
RED = (255, 0, 0) # Red color - RED
BLUE = (0, 0, 255) # Blue color - BLUE
GREEN = (0, 255, 0) # Green color - GREEN
FLASH_COLORS = [(255, 255, 255), (255, 0, 0)] # Colors used for flashing effect - FLASH COLORS
FPS = 30 # Frames per second - FRAMES PER SECOND
SCORE_HEIGHT = 40  # Height of the scoreboard - SCOREBOARD HEIGHT

# ---------------- Create Window ---------------- #
# Create window - CREATE THE GAME WINDOW
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Create the screen - CREATE SCREEN
pygame.display.set_caption("Maze Game") # Set the window title - WINDOW TITLE

# ---------------- Font for Scoreboard ---------------- #
# Font for scoreboard - SCOREBOARD FONT
font = pygame.font.Font(None, 36) # Create a font object - CREATE FONT
score = 0 # Initialize score - SCORE

# ---------------- Maze Layout ---------------- #
# 40x30 Maze (Fully Enclosed with Outer Corridor Free) - MAZE LAYOUT
maze = [
    "1111111111111111111111111111111111111111", # Maze row - ROW 1
    "1000000000000000000000000000000000000001", # Maze row - ROW 2
    "1011111011111111101111111011111110111101", # Maze row - ROW 3
    "1010000010000000100000001000000010000001", # Maze row - ROW 4
    "1010111111101110111111101101111011101101", # Maze row - ROW 5
    "1010000000001110000000001101000000001101", # Maze row - ROW 6
    "1011101111101110111111101101111011101101", # Maze row - ROW 7
    "1010001000000000100000001000000010000001", # Maze row - ROW 8
    "1011101011111110101111111011111010111101", # Maze row - ROW 9
    "1010000000000000000000000000000000000101", # Maze row - ROW 10
    "1011111011111111101111111011111110111101", # Maze row - ROW 11
    "1010000010000000100000001001000010000101", # Maze row - ROW 12
    "1010111111101110111111101101111011100101", # Maze row - ROW 13
    "1010000000001110000000001101000000000101", # Maze row - ROW 14
    "1011101111101110111111101101111011100101", # Maze row - ROW 15
    "1010001000000000100000001000000010000101", # Maze row - ROW 16
    "1011101011111110101111111011111010111101", # Maze row - ROW 17
    "1010000000000000000000000000000000000101", # Maze row - ROW 18
    "1011111011111111101111111011111110111101", # Maze row - ROW 19
    "1000000000000000000000000000000000000001", # Maze row - ROW 20
    "1011111111111111111111111111111111111101", # Maze row - ROW 21
    "1010001000000000100000001000000010000101", # Maze row - ROW 22
    "1011101011111110101111111011111010111101", # Maze row - ROW 23
    "1010000000000000000000000000000000000101", # Maze row - ROW 24
    "1011111011111111101111111011111110111101", # Maze row - ROW 25
    "1000000000000000000000000000000000000001", # Maze row - ROW 26
    "1111111111111111111111111111111111111111"  # Maze row - ROW 27
]

# ---------------- Ball Class ---------------- #
# Ball class - BALL CLASS (PLAYER)
class Ball:
    """Represents the player-controlled ball in the maze."""
    def __init__(self, x, y):
        """Initialize the ball with a given x and y coordinate."""
        self.x = x # X-coordinate of the ball - X POSITION
        self.y = y # Y-coordinate of the ball - Y POSITION
        self.speed = TILE_SIZE # Speed of the ball - SPEED

    def move(self, keys):
        """Move the ball based on pressed keys."""
        if keys[pygame.K_LEFT] and not check_collision(self.x - self.speed, self.y): # Move left if left key is pressed and no collision - MOVE LEFT
            self.x -= self.speed # Update x-coordinate - UPDATE X
        if keys[pygame.K_RIGHT] and not check_collision(self.x + self.speed, self.y): # Move right if right key is pressed and no collision - MOVE RIGHT
            self.x += self.speed # Update x-coordinate - UPDATE X
        if keys[pygame.K_UP] and not check_collision(self.x, self.y - self.speed): # Move up if up key is pressed and no collision - MOVE UP
            self.y -= self.speed # Update y-coordinate - UPDATE Y
        if keys[pygame.K_DOWN] and not check_collision(self.x, self.y + self.speed): # Move down if down key is pressed and no collision - MOVE DOWN
            self.y += self.speed # Update y-coordinate - UPDATE Y

    def draw(self):
        """Draw the ball on the screen."""
        pygame.draw.circle(screen, YELLOW, (self.x + BALL_RADIUS, self.y + BALL_RADIUS), BALL_RADIUS) # Draw the circle - DRAW CIRCLE

# ---------------- Check Collision Function ---------------- #
# Check collision with walls - CHECK COLLISION
def check_collision(x, y):
    """Check if the given coordinates collide with a wall in the maze."""
    col = x // TILE_SIZE # Calculate the column - COLUMN
    row = (y - SCORE_HEIGHT) // TILE_SIZE # Calculate the row - ROW
    if row < 0 or row >= len(maze) or col < 0 or col >= len(maze[row]): # Check for out of bounds - CHECK BOUNDS
        return True # Return True if out of bounds - OUT OF BOUNDS
    return maze[row][col] == "1" # Return True if there is a wall at the given coordinates - CHECK FOR WALL

# ---------------- Small Balls (Collectibles) ---------------- #
# Small balls (collectibles) - SMALL BALLS
small_balls = {(col * TILE_SIZE + BALL_RADIUS, row * TILE_SIZE + BALL_RADIUS + SCORE_HEIGHT) # Set comprehension to generate ball positions - GENERATE POSITIONS
               for row in range(len(maze)) for col in range(len(maze[row])) if maze[row][col] == "0"} # Check if maze cell is empty - CHECK FOR EMPTY CELL

# ---------------- Enemy Class ---------------- #
# Enemy class - ENEMY CLASS
class Enemy:
    """Represents an enemy in the maze."""
    def __init__(self, x, y):
        """Initialize the enemy with a given x and y coordinate."""
        self.x = x # X-coordinate of the enemy - X POSITION
        self.y = y # Y-coordinate of the enemy - Y POSITION
        self.vulnerable = False # Vulnerability status - VULNERABLE STATUS
        self.flash_timer = 0 # Timer for the flashing effect - FLASH TIMER
        self.color_index = 0 # Index for the color flashing effect - COLOR INDEX

    def move(self):
        """Move the enemy randomly."""
        directions = [(TILE_SIZE, 0), (-TILE_SIZE, 0), (0, TILE_SIZE), (0, -TILE_SIZE)] # Possible movement directions - DIRECTIONS
        dx, dy = random.choice(directions) # Choose a random direction - CHOOSE DIRECTION
        if not check_collision(self.x + dx, self.y + dy): # Check if the move is valid - CHECK COLLISION
            self.x += dx # Update x-coordinate - UPDATE X
            self.y += dy # Update y-coordinate - UPDATE Y

    def draw(self):
        """Draw the enemy on the screen."""
        color = FLASH_COLORS[self.color_index] if self.vulnerable else RED # Determine the color of the enemy - DETERMINE COLOR
        pygame.draw.rect(screen, color, (self.x, self.y, TILE_SIZE, TILE_SIZE)) # Draw the rectangle - DRAW RECTANGLE

# ---------------- Initialize Player and Enemies ---------------- #
# Initialize player and enemies - INITIALIZE OBJECTS
player = Ball(TILE_SIZE, TILE_SIZE + SCORE_HEIGHT) # Create a ball object for the player - CREATE PLAYER
enemies = [Enemy(200, 200 + SCORE_HEIGHT), Enemy(600, 200 + SCORE_HEIGHT), Enemy(200, 400 + SCORE_HEIGHT)] # Create enemy objects - CREATE ENEMIES

# ---------------- Game Loop ---------------- #
# Game loop - GAME LOOP
clock = pygame.time.Clock() # Create a clock object - CREATE CLOCK
running = True # Game loop flag - RUNNING FLAG
while running: # Game loop - GAME LOOP
    screen.fill(BLACK) # Fill the screen with black - DRAW BACKGROUND
    
    # Draw scoreboard - DRAW SCOREBOARD
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SCORE_HEIGHT)) # Draw a black rectangle for the scoreboard - DRAW RECTANGLE
    score_text = font.render(f"Score: {score}", True, WHITE) # Render the score text - RENDER SCORE
    screen.blit(score_text, (10, 10)) # Draw the score text on the screen - DRAW SCORE
    
    # Draw maze - DRAW MAZE
    for row in range(len(maze)): # Iterate over the rows of the maze - ITERATE ROWS
        for col in range(len(maze[row])): # Iterate over the columns of the maze - ITERATE COLUMNS
            if maze[row][col] == "1": # If the current cell is a wall - CHECK FOR WALL
                pygame.draw.rect(screen, BLUE, (col * TILE_SIZE, row * TILE_SIZE + SCORE_HEIGHT, TILE_SIZE, TILE_SIZE)) # Draw the wall - DRAW WALL
    
    # Event handling - HANDLE EVENTS
    for event in pygame.event.get(): # Iterate over the events - ITERATE EVENTS
        if event.type == pygame.QUIT: # If the user clicked the close button - CHECK FOR QUIT
            running = False # Set the running flag to False - STOP RUNNING
    
    # Player movement - PLAYER MOVEMENT
    keys = pygame.key.get_pressed() # Get the state of the keyboard - GET KEYBOARD STATE
    player.move(keys) # Move the player - MOVE PLAYER
    
    # Check collision with small balls - CHECK FOR BALL COLLISION
    if (player.x + BALL_RADIUS, player.y + BALL_RADIUS) in small_balls: # If the player collided with a small ball - CHECK FOR COLLISION
        small_balls.remove((player.x + BALL_RADIUS, player.y + BALL_RADIUS)) # Remove the small ball from the set - REMOVE BALL
        score += 10  # Increase score when collecting a small ball - INCREASE SCORE
    
    # Enemy movement - ENEMY MOVEMENT
    for enemy in enemies: # Iterate over the enemies - ITERATE ENEMIES
        enemy.move() # Move the enemy - MOVE ENEMY
    
    # Draw objects - DRAW OBJECTS
    player.draw() # Draw the player - DRAW PLAYER
    for enemy in enemies: # Iterate over the enemies - ITERATE ENEMIES
        enemy.draw() # Draw the enemy - DRAW ENEMY
    for sb in small_balls: # Iterate over the small balls - ITERATE SMALL BALLS
        pygame.draw.circle(screen, GREEN, sb, 5) # Draw the small ball - DRAW SMALL BALL
    
    # Update display - UPDATE DISPLAY
    pygame.display.flip() # Update the entire display - UPDATE THE DISPLAY
    clock.tick(FPS) # Limit the frame rate - LIMIT FRAME RATE

pygame.quit() # Quit pygame - QUIT PYGAME
