import gym
from gym.envs.registration import register
from env_python import UR5Env
# def register_solar_panel_env():
#     # Register the environment with gym
#     register(
#         id='SolarPanel-v0',  # The unique ID for your environment
#         entry_point='mujoco_solar:SolarPanelEnv',  # Path to your environment class
#         max_episode_steps=200,  # Set the maximum steps per episode
#     )

# # Call this function to register the environment
# register_solar_panel_env()

# After registering, you can create the environment like this:
# env = gym.make('SolarPanel-v0')
env = UR5Env()
obs = env.reset()
done = False
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
