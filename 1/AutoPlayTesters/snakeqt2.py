import random # Import the random module
import numpy as np # Import the NumPy module
import pygame # Import the Pygame module
import tensorflow as tf # Import the TensorFlow module
from tensorflow.keras.models import Sequential, load_model # Import Sequential and load_model from TensorFlow Keras
from tensorflow.keras.layers import Dense, Flatten, Input  # Import Dense, Flatten, Input layers from TensorFlow Keras
from tensorflow.keras.optimizers import Adam # Import the Adam optimizer from TensorFlow Keras
from collections import deque # Import deque from the collections module
import os  # Import the 'os' module
import multiprocessing as mp  # Import multiprocessing

# ---------------- Constants ---------------- #
GRID_WIDTH = 800  # Original Grid Size - DEFINE THE WIDTH OF THE GRID
GRID_HEIGHT = 600 # DEFINE THE HEIGHT OF THE GRID
GRID_SIZE = 10 # DEFINE THE SIZE OF EACH GRID CELL
GRID_COLOR = (0, 0, 0) # DEFINE THE COLOR OF THE GRID - BLACK
SNAKE_COLOR = (0, 255, 0) # DEFINE THE COLOR OF THE SNAKE - GREEN
APPLE_COLOR = (255, 0, 0) # DEFINE THE COLOR OF THE APPLE - RED
SCORE_COLOR = (255, 255, 255) # DEFINE THE COLOR OF THE SCORE TEXT - WHITE
FONT_SIZE = 14 # DEFINE THE FONT SIZE FOR THE SCORE TEXT
FONT_NAME = 'Arial' # DEFINE THE FONT NAME FOR THE SCORE TEXT
DEFAULT_FPS = 60 # DEFINE THE DEFAULT FRAMES PER SECOND
INITIAL_SNAKE_LENGTH = 2 #Reduced Snake Length - DEFINE THE INITIAL LENGTH OF THE SNAKE
GRID_COLS = GRID_WIDTH // GRID_SIZE # CALCULATE THE NUMBER OF COLUMNS IN THE GRID
GRID_ROWS = GRID_HEIGHT // GRID_SIZE # CALCULATE THE NUMBER OF ROWS IN THE GRID

# ---------------- Abstract Game Class ---------------- #
class AbstractGame:  # Define AbstractGame
    """
    Abstract base class for game environments.
    """
    def get_state(self):
        """
        Returns the current state of the game.
        """
        raise NotImplementedError # MUST BE IMPLEMENTED BY SUBCLASSES

    def step(self, action):
        """
        Performs an action in the game and returns the new state, reward, and done flag.
        """
        raise NotImplementedError # MUST BE IMPLEMENTED BY SUBCLASSES

    def reset(self):
        """
        Resets the game to its initial state.
        """
        raise NotImplementedError # MUST BE IMPLEMENTED BY SUBCLASSES

    def is_done(self):
        """
        Returns True if the game is over, False otherwise.
        """
        raise NotImplementedError # MUST BE IMPLEMENTED BY SUBCLASSES

    def get_action_space(self):
        """
        Returns the number of possible actions in the game.
        """
        raise NotImplementedError # MUST BE IMPLEMENTED BY SUBCLASSES

    def draw(self):  # Add draw to abstract class
        """
        Draws the current game state on the screen.
        """
        raise NotImplementedError # MUST BE IMPLEMENTED BY SUBCLASSES

    def quit(self):
        """
        Quits the game.
        """
        raise NotImplementedError # MUST BE IMPLEMENTED BY SUBCLASSES

# ---------------- Snake Game Class ---------------- #
class SnakeGame(AbstractGame):
    """
    Implements the Snake game environment.
    """
    def __init__(self, graphical=True, training_mode=False):  # Add graphical parameter and training mode
        """
        Initializes the Snake game.

        Parameters:
        - graphical: If True, the game will be displayed graphically.
        - training_mode: If True, the game will be run in training mode (headless).
        """
        self.graphical = graphical # STORE THE GRAPHICAL MODE SETTING
        if self.graphical: #Only initilize pygame if its graphical - AVOID ERRORS IN HEADLESS MODE
            os.environ['SDL_VIDEODRIVER'] = 'x11' #Force x11 video driver in linux - FIX FOR SOME LINUX SYSTEMS
            pygame.init() # INITIALIZE PYGAME
            try:
                self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT)) # CREATE THE GAME SCREEN
                pygame.display.set_caption("Snake Game - AI Learning") # SET THE WINDOW TITLE
                self.clock = pygame.time.Clock() # CREATE A CLOCK OBJECT
                self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE) # CREATE A FONT OBJECT
            except Exception as e:
                print(f"Error during Pygame initialization: {e}") # PRINT ANY ERRORS DURING PYGAME INITIALIZATION
                raise # RE-RAISE THE EXCEPTION
        else:
            self.screen = None  # No screen in headless mode - SET SCREEN TO NONE
            self.clock = None # SET CLOCK TO NONE
            self.font = None # SET FONT TO NONE

        self.state_size = 6 # DEFINE THE STATE SIZE
        self.action_size = 4 # DEFINE THE ACTION SIZE
        self.training_mode = training_mode #Save trainign mode parameter - STORE THE TRAINING MODE SETTING

        # Independent speed settings
        self.fps = DEFAULT_FPS # SET THE FRAMES PER SECOND
        self.snake_speed = 5  # Snake moves once every 'snake_speed' frames - CONTROL HOW OFTEN THE SNAKE MOVES
        self.move_counter = 0  # Counter to track movement intervals - KEEP TRACK OF MOVEMENT

        self.reset() # RESET THE GAME STATE
        if self.graphical: # DRAW THE GAME IF IN GRAPHICAL MODE
            self.draw() # DRAW THE GAME

    def get_screen_size(self):
        """
        Returns the screen size as a tuple (width, height, RGB channels).
        """
        return (GRID_WIDTH, GRID_HEIGHT, 3)  # (width, height, RGB channels)

    def get_screen(self):
        """
        Captures the game screen as a NumPy array for CNN input.
        """
        if self.graphical: # ONLY CAPTURE THE SCREEN IF IN GRAPHICAL MODE
            screen_array = pygame.surfarray.array3d(pygame.display.get_surface()) # GET THE SCREEN AS A NUMPY ARRAY
            return np.transpose(screen_array, (1, 0, 2))  # Convert Pygame format to standard (H, W, C) - TRANSPOSE THE ARRAY
        else:
            return None # no screen in headless mode - RETURN NONE IF IN HEADLESS MODE

    def reset(self):
        """
        Resets the game state.
        """
        self.snake = [(GRID_COLS // 2, GRID_ROWS // 2)] # INITIALIZE THE SNAKE AT THE CENTER OF THE GRID
        for i in range(1, INITIAL_SNAKE_LENGTH):
            self.snake.append((self.snake[0][0] - i, self.snake[0][1])) # ADD SEGMENTS TO THE SNAKE

        self.direction = (1, 0) # SET THE INITIAL DIRECTION OF THE SNAKE - RIGHT
        self.apples = [] # INITIALIZE THE LIST OF APPLES
        # Initialize with a single apple in a valid position
        while not self.apples:  # Ensure at least one valid apple - MAKE SURE THERE IS AT LEAST ONE APPLE
            apple_x = random.randint(0, GRID_COLS - 1) # GENERATE A RANDOM X COORDINATE FOR THE APPLE
            apple_y = random.randint(0, GRID_ROWS - 1) # GENERATE A RANDOM Y COORDINATE FOR THE APPLE
            if (apple_x, apple_y) not in self.snake: # CHECK IF THE APPLE IS NOT ON THE SNAKE
                self.apples.append((apple_x, apple_y)) # ADD THE APPLE TO THE LIST


        self.generate_apples() #Create more apples on top - GENERATE MORE APPLES
        self.score = 0 # RESET THE SCORE
        self.apples_eaten = 0 # RESET THE NUMBER OF APPLES EATEN
        self.game_over = False # SET THE GAME OVER FLAG TO FALSE
        self.fibonacci_rewards = [10, 10] # INITIALIZE THE FIBONACCI REWARDS LIST
        self.move_counter = 0  # Reset movement counter - RESET THE MOVEMENT COUNTER
        return self.get_state() #Ensure to give a state at the end of the reset - RETURN THE INITIAL STATE

    def generate_apples(self):
        """
        Generates more apples if there are less than 30 on the screen.
        """
        # Ensure there are always 30 apples available - MAINTAIN A CONSTANT NUMBER OF APPLES

        while len(self.apples) < 30: # CHECK IF THERE ARE LESS THAN 30 APPLES
            apple_position = self.generate_single_apple() # GENERATE A SINGLE APPLE
            if apple_position not in self.apples: # CHECK IF THE APPLE IS NOT ALREADY IN THE LIST
                self.apples.append(apple_position) # ADD THE APPLE TO THE LIST

    def generate_single_apple(self):
        """
        Generates a single apple at a random location that is not occupied by the snake.
        """
        while True: # INFINITE LOOP UNTIL A VALID APPLE POSITION IS FOUND
            apple_x = random.randint(0, GRID_COLS - 1) # GENERATE A RANDOM X COORDINATE FOR THE APPLE
            apple_y = random.randint(0, GRID_ROWS - 1) # GENERATE A RANDOM Y COORDINATE FOR THE APPLE
            apple_position = (apple_x, apple_y) # CREATE A TUPLE WITH THE APPLE POSITION

            if apple_position not in self.snake: # CHECK IF THE APPLE IS NOT ON THE SNAKE
                return apple_position # RETURN THE APPLE POSITION


    def get_state(self):
        """
        Returns the current state of the game as a NumPy array.
        """
        head_x, head_y = self.snake[0] # GET THE COORDINATES OF THE SNAKE'S HEAD

        # Handle case where there are no apples - AVOID ERRORS WHEN THERE ARE NO APPLES
        if not self.apples:
            # Return a default state filled with zeros and correct dtype - RETURN A ZEROED ARRAY
            return np.zeros(self.state_size, dtype=np.float32)

        try:
            apple_x, apple_y = self.apples[0] # GET THE COORDINATES OF THE FIRST APPLE
        except IndexError:  # Handle the case if self.apples is empty - AVOID ERRORS WHEN THERE ARE NO APPLES
            #print("ERROR: self.apples is empty! Returning default state.") #Commented out
            return np.zeros(self.state_size, dtype=np.float32) # RETURN A ZEROED ARRAY

        distance_to_left = head_x / GRID_COLS # CALCULATE THE DISTANCE TO THE LEFT WALL - NORMALIZED
        distance_to_right = (GRID_COLS - 1 - head_x) / GRID_COLS # CALCULATE THE DISTANCE TO THE RIGHT WALL - NORMALIZED
        distance_to_up = head_y / GRID_ROWS # CALCULATE THE DISTANCE TO THE TOP WALL - NORMALIZED
        distance_to_down = (GRID_ROWS - 1 - head_y) / GRID_ROWS # CALCULATE THE DISTANCE TO THE BOTTOM WALL - NORMALIZED

        rel_x = (apple_x - head_x) / GRID_COLS # CALCULATE THE RELATIVE X COORDINATE OF THE APPLE - NORMALIZED
        rel_y = (apple_y - head_y) / GRID_COLS # CALCULATE THE RELATIVE Y COORDINATE OF THE APPLE - NORMALIZED

        # Ensure correct dtype and shape - MAKE SURE THE ARRAY HAS THE CORRECT DATA TYPE AND SHAPE
        state = np.array([rel_x, rel_y, distance_to_left, distance_to_right, distance_to_up, distance_to_down], dtype=np.float32) # CREATE THE STATE ARRAY
        assert state.shape == (self.state_size,), f"State shape is incorrect! Expected {(self.state_size,)}, got {state.shape}" # CHECK IF THE SHAPE IS CORRECT
        return state # RETURN THE STATE ARRAY


    def step(self, action):
        """
        Updates the game state based on the given action.
        Snake only moves if enough frames have passed.
        """
        if self.graphical: # CHECK IF THE GAME IS IN GRAPHICAL MODE
            self.clock.tick(self.fps)  # Maintain constant FPS - LIMIT THE FRAMES PER SECOND
        self.move_counter += 1  # Increment movement counter - UPDATE THE MOVEMENT COUNTER

        # Move the snake only if enough frames have passed - CONTROL THE SNAKE'S SPEED
        if self.move_counter < self.snake_speed:
            return -1, self.game_over  # Minor penalty for waiting - GIVE A PENALTY FOR WAITING

        self.move_counter = 0  # Reset counter - RESET THE MOVEMENT COUNTER

        # Convert action index to movement direction - CONVERT THE ACTION TO A DIRECTION
        if action == 0:
            new_direction = (0, -1) # UP
        elif action == 1:
            new_direction = (0, 1) # DOWN
        elif action == 2:
            new_direction = (-1, 0) # LEFT
        elif action == 3:
            new_direction = (1, 0) # RIGHT
        else:
            new_direction = self.direction # NO CHANGE

        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction # UPDATE THE DIRECTION

        head_x, head_y = self.snake[0] # GET THE COORDINATES OF THE SNAKE'S HEAD
        new_head = (head_x + self.direction[0], head_y + self.direction[1]) # CALCULATE THE NEW HEAD POSITION

        if new_head[0] < 0 or new_head[0] >= GRID_COLS or new_head[1] < 0 or new_head[1] >= GRID_ROWS:
            self.game_over = True # SET THE GAME OVER FLAG TO TRUE
            return -1000, self.game_over  # HUGE PENALTY FOR DYING - GIVE A HUGE PENALTY

        if new_head in self.snake[1:]:
            self.game_over = True # SET THE GAME OVER FLAG TO TRUE
            return -1000, self.game_over  # HUGE PENALTY FOR DYING - GIVE A HUGE PENALTY

        self.snake.insert(0, new_head) # INSERT THE NEW HEAD INTO THE SNAKE LIST

        if new_head in self.apples:
            self.apples.remove(new_head) # REMOVE THE APPLE FROM THE LIST
            self.apples.append(self.generate_single_apple()) # ADD A NEW APPLE TO THE LIST
            self.apples_eaten += 1 # INCREMENT THE NUMBER OF APPLES EATEN
            self.score += 1 # INCREMENT THE SCORE
            reward = 100 # Huge reward for eating apple - GIVE A HUGE REWARD
            self.fibonacci_rewards.append(reward) # ADD THE REWARD TO THE FIBONACCI REWARDS LIST
            return reward, self.game_over # RETURN THE REWARD AND THE GAME OVER FLAG

        self.snake.pop() # REMOVE THE LAST SEGMENT OF THE SNAKE
        return -1, self.game_over #Penalty For surviving - GIVE A PENALTY FOR SURVIVING

    def draw(self):
        """
        Renders the game state visually using Pygame.
        """
        if self.graphical: # CHECK IF THE GAME IS IN GRAPHICAL MODE
            self.screen.fill(GRID_COLOR) # FILL THE SCREEN WITH THE GRID COLOR

            for x, y in self.snake:
                pygame.draw.rect(self.screen, SNAKE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)) # DRAW THE SNAKE

            for x, y in self.apples:
                pygame.draw.rect(self.screen, APPLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)) # DRAW THE APPLES

            score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR) # RENDER THE SCORE TEXT
            self.screen.blit(score_text, (10, 10)) # BLIT THE SCORE TEXT TO THE SCREEN

            pygame.display.flip() # UPDATE THE DISPLAY

    def quit(self):
        """
        Quits the game.
        """
        if self.graphical: # CHECK IF THE GAME IS IN GRAPHICAL MODE
            pygame.quit() # QUIT PYGAME

    def is_done(self):
        """
        Returns True if the game is over, False otherwise.
        """
        return self.game_over # RETURN THE GAME OVER FLAG

    def get_action_space(self):
        """
        Returns the number of possible actions.
        """
        return self.action_size # RETURN THE ACTION SIZE

# ---------------- DQN Agent Class ---------------- #
class DQN_PER_Agent:
    """
    Deep Q-Network (DQN) agent with Prioritized Experience Replay (PER)
    """
    def __init__(self, state_size, action_size, model_path=None, epsilon_decay=0.999, epsilon_min = 0.01):
        """
        Initializes the DQN agent with PER.
        """
        self.state_size = state_size  # Number of state variables - DEFINE THE STATE SIZE
        self.action_size = action_size  # Number of possible actions - DEFINE THE ACTION SIZE
        self.gamma = 0.99  # Discount factor - HOW MUCH TO VALUE FUTURE REWARDS
        self.epsilon = 1.0  # Exploration rate - HOW OFTEN TO EXPLORE
        self.epsilon_min = epsilon_min  # Minimum exploration rate - THE LOWEST EXPLORATION RATE
        self.epsilon_decay = epsilon_decay  # DECAY PER STEP - ADJUSTED - HOW QUICKLY TO STOP EXPLORING
        self.learning_rate = 0.001  # Learning rate - HOW QUICKLY TO LEARN
        self.batch_size = 10000  # Reduced Batch Size - HOW MANY SAMPLES TO LEARN FROM
        self.memory_size = 100000  # Replay buffer size - HOW MUCH MEMORY TO USE
        self.replay_buffer = deque(maxlen=self.memory_size)  # Experience replay memory - STORE EXPERIENCES
        self.alpha = 0.6  # Prioritization exponent - HOW MUCH TO PRIORITIZE EXPERIENCES
        self.beta = 0.4  # Importance sampling exponent - HOW MUCH TO CORRECT FOR BIAS
        self.beta_increment = 0.001  # Beta increment per step - HOW QUICKLY TO CORRECT FOR BIAS
        self.priorities = deque(maxlen=self.memory_size)  # Stores priority values - STORE PRIORITIES
        self.reward_history = deque(maxlen=100) # Store the rewards in the last 100 episodes - TRACK RECENT REWARDS
        self.epsilon_reset_threshold = 10 # If reward does not improve after 10 episodes - WHEN TO RESET EXPLORATION

        # Initialize the neural network model - CREATE THE MODEL
        self.model = self._build_model() # BUILD THE MODEL
        self.target_model = self._build_model() # BUILD THE TARGET MODEL
        self.update_target_model() # UPDATE THE TARGET MODEL

        if model_path: # CHECK IF A MODEL PATH IS PROVIDED
            self.load(model_path)  # Load existing model if available - LOAD THE MODEL

    def _build_model(self):
        """Builds a simple neural network for DQN."""
        model = Sequential([
            Input(shape=(self.state_size,)),  # Use Input layer for defining the shape - DEFINE THE INPUT LAYER
            Flatten(), # FLATTEN THE INPUT
            Dense(32, activation='relu'), # FIRST DENSE LAYER
            Dense(32, activation='relu'), # SECOND DENSE LAYER
            Dense(32, activation='relu'), # THIRD DENSE LAYER
            Dense(32, activation='relu'), # FOURTH DENSE LAYER
            Dense(32, activation='relu'), # FIFTH DENSE LAYER
            Dense(32, activation='relu'), # SIXTH DENSE LAYER
            Dense(self.action_size, activation='linear') # OUTPUT LAYER
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate)) # COMPILE THE MODEL
        return model # RETURN THE MODEL

    def update_target_model(self):
        """Copies the weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights()) # COPY THE WEIGHTS

    def get_action(self, state):
        """Returns an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon: # CHECK IF WE SHOULD EXPLORE
            return random.randrange(self.action_size) # EXPLORE - RETURN A RANDOM ACTION

        # Add these lines:
        state = np.asarray(state, dtype=np.float32) # Ensure correct dtype - ENSURE THE STATE HAS THE CORRECT DATA TYPE
        state = np.ascontiguousarray(state) # Ensure it's contiguous in memory - ENSURE THE STATE IS CONTIGUOUS IN MEMORY
        #print(f"State shape: {state.shape}, State: {state}")  # ADD THIS LINE
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0) # PREDICT THE Q-VALUES FOR EACH ACTION
        return np.argmax(q_values[0]) # RETURN THE ACTION WITH THE HIGHEST Q-VALUE

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in replay memory with priority."""
        if not isinstance(state, np.ndarray) or state.shape != (self.state_size,): # CHECK IF THE STATE IS VALID
            print("WARNING: Discarding experience with invalid state shape!") # PRINT A WARNING
            return # Don't store it! - DO NOT STORE THE EXPERIENCE

        priority = max(self.priorities, default=1.0)  # Use max priority if memory is empty - SET THE PRIORITY TO THE MAXIMUM
        self.replay_buffer.append((state, action, reward, next_state, done)) # APPEND THE EXPERIENCE TO THE REPLAY BUFFER
        self.priorities.append(priority) # APPEND THE PRIORITY TO THE LIST
        self.reward_history.append(reward) # Store the reward - STORE THE REWARD

    def replay(self):
        """Trains the network using Prioritized Experience Replay."""
        if len(self.replay_buffer) < self.batch_size: # CHECK IF THERE ARE ENOUGH EXPERIENCES IN THE REPLAY BUFFER
            return # DO NOT TRAIN IF THERE ARE NOT ENOUGH EXPERIENCES

        priorities = np.array(self.priorities) # CONVERT THE PRIORITIES TO A NUMPY ARRAY
        probs = priorities ** self.alpha # CALCULATE THE PROBABILITIES
        probs /= probs.sum() # NORMALIZE THE PROBABILITIES

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs) # CHOOSE THE INDICES
        batch = [self.replay_buffer[i] for i in indices] # CREATE THE BATCH

        # Initialize states with an empty list in case the batch is empty - INITIALIZE THE STATES
        states, actions, rewards, next_states, dones = [], [], [], [], [] # INITIALIZE THE LISTS
        if batch: # CHECK IF THE BATCH IS NOT EMPTY
            states, actions, rewards, next_states, dones = zip(*batch) # UNZIP THE BATCH

        states = np.array(states, dtype=np.float32)  # Enforce dtype here! - ENSURE THE STATES HAVE THE CORRECT DATA TYPE
        next_states = np.array(next_states, dtype=np.float32) #Enforce dtype here! - ENSURE THE NEXT STATES HAVE THE CORRECT DATA TYPE
        #print("TensorFlow version:", tf.__version__)

        target_q_values = self.model.predict(states, verbose=0) # PREDICT THE TARGET Q-VALUES
        #print(f"Shape of target_q_values: {target_q_values.shape}")
        next_q_values = self.target_model.predict(next_states, verbose=0) # PREDICT THE NEXT Q-VALUES
        #print(f"Shape of next_q_values: {next_q_values.shape}")

        for i, index in enumerate(indices):
            if dones[i]: # CHECK IF THE EPISODE IS DONE
                target_q_values[i][actions[i]] = rewards[i] # SET THE TARGET Q-VALUE TO THE REWARD
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i]) # CALCULATE THE TARGET Q-VALUE

            # Update priority - UPDATE THE PRIORITY
            self.priorities[index] = abs(rewards[i] + self.gamma * np.max(next_q_values[i]) - target_q_values[i][actions[i]]) + 1e-5 # CALCULATE THE NEW PRIORITY

        self.model.fit(states, target_q_values, epochs=1, verbose=0, batch_size=self.batch_size) # TRAIN THE MODEL

        # Update epsilon (exploration rate) - PER STEP - UPDATE THE EXPLORATION RATE
        if self.epsilon > self.epsilon_min: # CHECK IF THE EXPLORATION RATE IS GREATER THAN THE MINIMUM
            self.epsilon *= self.epsilon_decay # DECAY THE EXPLORATION RATE

    def save(self, model_path):
        """Saves the trained model."""
        self.model.save(model_path) # SAVE THE MODEL
        print(f"Model saved at {model_path}") # PRINT A MESSAGE

    def load(self, model_path):
        """Loads a pre-trained model if available."""
        try:
            self.model = tf.keras.models.load_model(model_path) # LOAD THE MODEL
            self.update_target_model() # UPDATE THE TARGET MODEL
            print(f"Model loaded from {model_path}") # PRINT A MESSAGE
        except Exception as e:
            print(f"Error loading model: {e}") # PRINT AN ERROR MESSAGE
            #If model doesn't exist, create one
            self.model = self._build_model() # BUILD THE MODEL
            self.target_model = self._build_model() # BUILD THE TARGET MODEL
            self.update_target_model() # UPDATE THE TARGET MODEL
            print(f"Creating new model at {model_path}") # PRINT A MESSAGE
            self.save(model_path) # SAVE THE MODEL

    def check_and_reset_epsilon(self):
        """Resets epsilon to 1.0 if reward does not improve after a threshold of the last 10 episodes."""
        if len(self.reward_history) >= self.epsilon_reset_threshold: # Check if enough rewards are in history - NEED ENOUGH DATA
            # Calculate the mean reward over a recent window.
            mean_reward = sum(self.reward_history) / len(self.reward_history) # CALCULATE THE MEAN REWARD

            # Get the mean reward over the last self.epsilon_reset_threshold episodes.
            last_mean_reward = sum(list(self.reward_history)[-self.epsilon_reset_threshold:]) / self.epsilon_reset_threshold # CALCULATE THE LAST MEAN REWARD

            # Check if the mean reward over the last self.epsilon_reset_threshold episodes is less than the current reward
            if mean_reward < last_mean_reward: # CHECK IF THE REWARD IS NOT IMPROVING
                print("Resetting epsilon to 1.0 due to lack of reward improvement.") # PRINT A MESSAGE
                self.epsilon = 1.0 # RESET EPSILON

# ---------------- Training Function ---------------- #
def train_agent(game, agent, num_steps):
    """Trains the agent for a specified number of steps within the game."""
    state = game.reset() # RESET THE GAME
    total_reward = 0 # INITIALIZE THE TOTAL REWARD
    for step in range(num_steps): # ITERATE OVER THE NUMBER OF STEPS
        action = agent.get_action(state) # GET THE ACTION
        reward, done = game.step(action) # TAKE THE ACTION
        next_state = game.get_state() # GET THE NEXT STATE
        agent.remember(state, action, reward, next_state, done) # REMEMBER THE EXPERIENCE
        agent.replay() # TRAIN THE AGENT
        state = next_state # UPDATE THE STATE
        total_reward += reward # UPDATE THE TOTAL REWARD
        if done: # CHECK IF THE EPISODE IS DONE
            state = game.reset() # RESET THE GAME
    return total_reward # RETURN THE TOTAL REWARD

# ---------------- Function to Simulate Training ---------------- #
def simulate_training(model_path="model.keras", training_steps=500):
    """Simulates the training process to advance the agent to a later stage."""
    game = SnakeGame(graphical=False, training_mode=True)  # Headless training mode - CREATE THE GAME IN HEADLESS MODE
    agent = DQN_PER_Agent(game.state_size, game.action_size, model_path=model_path) # CREATE THE AGENT

    print(f"Simulating {training_steps} training steps...") # PRINT A MESSAGE
    train_agent(game, agent, training_steps) #TRAINING - TRAIN THE AGENT
    agent.save(model_path)  # Save the "trained" model - SAVE THE MODEL
    print("Simulated training complete.") # PRINT A MESSAGE
    game.quit() # QUIT THE GAME


# ---------------- Load Model & Run Game ---------------- #
def autoplay(model_path="model.keras"):
    """Plays the Snake game using a trained DQN agent."""
    game = SnakeGame()  # Graphical mode - CREATE THE GAME IN GRAPHICAL MODE

    agent = DQN_PER_Agent(game.state_size, game.action_size, model_path=model_path)  # Load model for playing - CREATE THE AGENT AND LOAD THE MODEL

    running = True # SET THE RUNNING FLAG TO TRUE
    while running: # WHILE THE GAME IS RUNNING
        state = game.get_state() # GET THE STATE
        action = agent.get_action(state) # GET THE ACTION
        _, game_over = game.step(action) # TAKE THE ACTION
        game.draw() # DRAW THE GAME

        if game_over: # CHECK IF THE GAME IS OVER
            game.reset() # RESET THE GAME

        for event in pygame.event.get(): # HANDLE EVENTS
            if event.type == pygame.QUIT: # CHECK IF THE USER WANTS TO QUIT
                running = False # SET THE RUNNING FLAG TO FALSE
                break # BREAK OUT OF THE LOOP

    game.quit() # QUIT THE GAME
    pygame.quit() # QUIT PYGAME
    return # RETURN

# --- Headless Training Script ---
def train_headless(model_path="model.keras"):
    """Trains the DQN agent on the Snake game in a headless environment."""

    # Simulate training for 500 steps and save the model
    print("Simulating 500 steps before training...")  # SPECIFIC MESSAGE - PRINT A MESSAGE
    simulate_training(model_path, 500) # SIMULATE TRAINING
    print("Simulated 500 steps complete, training will begin...")  # SPECIFIC MESSAGE - PRINT A MESSAGE

    # Initialize the game (headless)
    game = SnakeGame(graphical=False)  # Headless mode - CREATE THE GAME IN HEADLESS MODE

    state_size = game.state_size # GET THE STATE SIZE
    action_size = game.action_size # GET THE ACTION SIZE

    # Training parameters
    EPISODES = 500  # More episodes - DEFINE THE NUMBER OF EPISODES
    UPDATE_TARGET_FREQUENCY = 5  # Update target network every X episodes - DEFINE HOW OFTEN TO UPDATE THE TARGET NETWORK
    SAVE_MODEL_FREQUENCY = 25 #Save after every 25 episodes. - DEFINE HOW OFTEN TO SAVE THE MODEL

    epsilon_min = 0.01 # DEFINE THE MINIMUM EXPLORATION RATE
    epsilon_decay = 0.999  # Slower decay - DEFINE THE EXPLORATION DECAY

    # Initialize the agent - Load Model if Available - **DO THIS ONCE!**
    agent = DQN_PER_Agent(state_size, action_size, model_path=model_path, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min) # CREATE THE AGENT

    try:
        for episode in range(1, EPISODES + 1): # ITERATE OVER THE EPISODES
            state = game.reset()  # Reset the game at the beginning of each episode - RESET THE GAME
            done = False # SET THE DONE FLAG TO FALSE
            total_reward = 0 # INITIALIZE THE TOTAL REWARD

            while not done: # WHILE THE EPISODE IS NOT DONE
                # Agent selects an action
                action = agent.get_action(state) # GET THE ACTION

                # Agent performs the action and get next state and reward
                reward, done = game.step(action) # TAKE THE ACTION
                next_state = game.get_state() # GET THE NEXT STATE

                # Store the experience in replay memory
                agent.remember(state, action, reward, next_state, done) # REMEMBER THE EXPERIENCE

                # Train the agent
                agent.replay()  # EPSILON IS DECAYED HERE NOW, PER STEP - TRAIN THE AGENT

                state = next_state # UPDATE THE STATE
                total_reward += reward # UPDATE THE TOTAL REWARD

            # Adaptive Epsilon Reset
            agent.check_and_reset_epsilon() # CHECK AND RESET EPSILON

            # Save the model at the end of some episodes
            if episode % SAVE_MODEL_FREQUENCY == 0:
                agent.save(model_path) # Saving code - SAVE THE MODEL

            # Update target network
            if episode % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_model() # UPDATE THE TARGET MODEL
                print(f"Episode {episode}: Target model updated") # PRINT A MESSAGE

            # Print episode information
            print(f"Episode: {episode}/{EPISODES}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}, Apples Eaten: {game.apples_eaten}") # PRINT EPISODE INFORMATION

    except KeyboardInterrupt: # HANDLE KEYBOARD INTERRUPTS
        print("\nTraining interrupted. Saving model...") # PRINT A MESSAGE
        agent.save(model_path) # SAVE THE MODEL
    finally: # ALWAYS RUN THIS CODE
        game.quit() # QUIT THE GAME
