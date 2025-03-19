import numpy as np
import time
from OA_env import OA_env  # or whatever your environment file is named

def test_random_actions(num_episodes=2, max_steps=50):
    # Initialize the environment
    env = OA_env()  # or pass any constructor args needed
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            print(action)
            action = 4
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            env.render()
            
            steps += 1
            
        print(f"Episode {ep+1} ended. Steps: {steps}, Total Reward: {episode_reward:.2f}")
    
    env.close()

if __name__ == "__main__":
    test_random_actions(num_episodes=2, max_steps=50)
