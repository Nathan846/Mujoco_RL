import json
import numpy as np
import matplotlib.pyplot as plt
import re
from OA_env import OA_env

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
    actions = [entry["discrete_action"] for entry in logs if "discrete_action" in entry]
    return actions

def get_step_size_from_filename(filename):
    match = re.search(r'p(\d+)', filename)
    if match:
        step_str = match.group(1)
        return float("0." + step_str) if len(step_str) > 1 else float(step_str)
    return None

def test_trajectory_in_env(file_path):
    step_size = get_step_size_from_filename(file_path)
    init_info, trajectory = load_trajectory(file_path)
    actions = extract_discrete_actions_from_log(file_path)

    env = OA_env(render=False)
    obs = env.reset()

    rewards = []
    steps = []

    for step_count, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        if reward is None:
            continue
        rewards.append(reward)
        steps.append(step_count)
        if done:
            obs = env.reset()

    return steps, rewards, step_size

if __name__ == "__main__":
    file_list = [
        "place_91_updatedx2.json",
        "place_91_updated.json",
        "place_91_updatedx8.json",
        "place_91_updatedx4.json"
    ]

    plt.figure(figsize=(12, 6))

    for file in file_list:
        steps, rewards, step_size = test_trajectory_in_env(file)
        label = f"Step Size {step_size}"
        plt.plot(steps, rewards, label=label)

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward vs Step for Different Step Sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
