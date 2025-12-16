import random
import multiprocessing as mp
import numpy as np
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # Set before importing pygame
from snake import SnakeGame
from dqn_per_agent import DQN_PER_Agent

def run_game(seed, initial_weights):
    random.seed(seed)
    game = SnakeGame(render=False)
    agent = DQN_PER_Agent(game)
    
    if initial_weights is not None:
        agent.model.set_weights(initial_weights)
        agent.update_target_model()
    
    total_reward = 0
    state = game.get_state()
    
    for _ in range(1000):
        action = agent.get_action(state)
        reward, done = game.step(action)
        next_state = game.get_state()
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        if done:
            state = game.reset()
    
    game.quit()
    return total_reward, agent.model.get_weights()

if __name__ == "__main__":
    num_workers = 1
    
    temp_game = SnakeGame(render=False)
    temp_agent = DQN_PER_Agent(temp_game)
    initial_weights = temp_agent.model.get_weights()
    temp_game.quit()
    
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(run_game, [(i, initial_weights) for i in range(num_workers)])
    
    rewards, weights_list = zip(*results)
    avg_reward = sum(rewards) / num_workers
    print("Average reward:", avg_reward)
    
    avg_weights = [np.mean([w[i] for w in weights_list], axis=0) for i in range(len(initial_weights))]
    final_agent = DQN_PER_Agent(temp_game)
    final_agent.model.set_weights(avg_weights)
    final_agent.save("trained_model.h5")
