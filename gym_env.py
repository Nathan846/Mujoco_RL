import numpy as np
from env_main import MuJoCoEnv

# Initialize environment
env = MuJoCoEnv("ur10e/ur10e.xml")  # Update path to your MuJoCo XML model

obs = env.reset()
done = False

while not done:  # Run for 1000 timesteps or until done
    # Take a random action
    action = env.action_space.sample()  # Random action within action space
    obs, reward, done, info = env.step(action)

    # Render the environment
    env.render()

    # Print observations and reward for debugging
    print(f"Observation: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

    if done:
        print("Reached the target! Resetting environment.")
        obs = env.reset()

env.close()
