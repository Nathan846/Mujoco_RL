import numpy as np
import json
import os

def load_trajectory(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    joint_angles = [entry['joint_angles'] for entry in data['data']]
    return np.array(joint_angles)

def calculate_rmse(traj1_joint_angles, traj2_joint_angles):
    min_len = min(len(traj1_joint_angles), len(traj2_joint_angles))
    traj1_common = traj1_joint_angles[:min_len]
    traj2_common = traj2_joint_angles[:min_len]
    
    squared_diff = np.sum((traj1_common - traj2_common)**2, axis=1)
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    return rmse

first_file = 'trajectories/traj1.json'
traj1_joint_angles = load_trajectory(first_file)

folder_path = 'trajectories'
files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

for file in files:
    if file != os.path.basename(first_file):
        traj2_joint_angles = load_trajectory(os.path.join(folder_path, file))
        rmse = calculate_rmse(traj1_joint_angles, traj2_joint_angles)
        print(f"RMSE between {first_file} and {file}: {rmse}")
