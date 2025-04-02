import json
import numpy as np
from OA_env import OA_env
import pandas as pd
def load_trajectory(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    init_info = data["initial_values"]
    frame_list = data["data"]

    trajectory = [frame["joint_angles"] for frame in frame_list]
    return init_info, trajectory

def extract_discrete_actions_from_log(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    logs = data["data"]
    actions = []

    for entry in logs:
        action = entry.get("discrete_action", None)
        if action is not None:
            actions.append(action)

    return actions
def test_trajectory_in_env(file_path, step_size=0.01):
    init_info, trajectory = load_trajectory(file_path)
    actions = extract_discrete_actions_from_log(file_path)
    env = OA_env()
    obs = env.reset()

    total_reward = 0.0
    step_count = 0
    
    print("Starting discrete trajectory replay ...")

    for action in actions:
        obs, reward, done, info = env.step(action)
        if(not reward):
            continue
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}, Action: {action}, Reward: {reward:.3f}")
        if done:
            print("Environment returned done=True. Resetting...")
            obs = env.reset()
    print(f"Replay finished. Total steps: {step_count}, Accumulated reward: {total_reward:.3f}")

    while True:
        continue

if __name__ == "__main__":
    file_path = "place_73_expanded.json"
    test_trajectory_in_env(file_path)
