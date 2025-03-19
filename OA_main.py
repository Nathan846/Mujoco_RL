import os
import glob
import json
import numpy as np
import torch
import random
from collections import deque
from option_critic import OptionCriticFeatures, actor_loss, critic_loss
from OA_env import OA_env
from copy import deepcopy
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, option, reward, next_obs, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, options, rewards, next_obs, dones = zip(*batch)
        obs = torch.FloatTensor(obs)
        next_obs = torch.FloatTensor(next_obs)
        return obs, options, rewards, next_obs, dones
    def __len__(self):
        return len(self.buffer)
class Args:
    gamma = 0.99
    termination_reg = 0.01
    entropy_reg = 0.01
def process_trajectory(trajectory_data):
    if isinstance(trajectory_data, list):
        data_points = trajectory_data
    else:
        data_points = trajectory_data.get("data", [])
    states = []
    actions = []
    options = []
    data_points = data_points[0]
    for i, point in enumerate(data_points):
        
        joint_angles = point.get("joint_angles", [])
        slab_position = point.get("slab_position", [])
        slab_orientation = point.get("slab_orientation", [])
        contact = point.get("contact", {})
        forces = contact.get("forces", {})
        normal_force = forces.get("normal_force", 0.0)
        state = np.array(joint_angles + slab_position + slab_orientation + [normal_force], dtype=np.float32)
        states.append(state)
        if i > 0:
            prev_joint_angles = np.array(data_points[i-1].get("joint_angles", []), dtype=np.float32)
            current_joint_angles = np.array(joint_angles, dtype=np.float32)
            action = current_joint_angles - prev_joint_angles
        else:
            action = np.zeros(len(joint_angles), dtype=np.float32)
        actions.append(action)
        options.append(0)
    return {"states": np.array(states),
            "actions": np.array(actions),
            "options": np.array(options)}

def load_trajectories_from_folder(folder_path):
    trajectories = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    print(f"Found {len(json_files)} trajectory files in {folder_path}")
    for file in json_files:
        with open(file, 'r') as f:
            traj_data = json.load(f)
            processed = process_trajectory(traj_data)
            trajectories.append(processed)
                # print(f"Error processing file {file}: {e}")
    return trajectories

def fill_buffer_with_expert_data(buffer, folder_path):
    trajectories = load_trajectories_from_folder(folder_path)
    for traj in trajectories:
        states = traj["states"]
        actions = traj["actions"]
        options = traj["options"]
        for i in range(len(states)-1):
            s = states[i]
            next_s = states[i+1]
            option = options[i]
            reward = 0.0
            done = False
            buffer.push(s, option, reward, next_s, done)

def train_option_critic_with_demos():
    device = "cpu"
    env = OA_env(device=device)
    obs_dim = env.obs_dim  # 53
    act_dim = env.action_space.n  # 14
    
    num_options = 2
    agent = OptionCriticFeatures(
        in_features=obs_dim,
        num_actions=act_dim,
        num_options=num_options,
        device=device
    )
    agent_prime = deepcopy(agent)
    
    buffer = ReplayBuffer(capacity=10000)
    fill_buffer_with_expert_data(buffer, "trajectories")
    
    optimizer = torch.optim.RMSprop(agent.parameters(), lr=0.0005)
    total_steps = 0
    num_episodes = 1000
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            state_t = agent.get_state(torch.FloatTensor(obs))
            epsilon = agent.epsilon
            if random.random() < epsilon:
                current_option = random.randint(0, num_options - 1)
            else:
                current_option = agent.greedy_option(state_t)
            action, logp, entropy = agent.get_action(state_t, current_option)
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            buffer.push(obs, current_option, reward, next_obs, done)
            
            if len(buffer) > 32 and total_steps % 4 == 0:
                batch = buffer.sample(32)
                c_loss = critic_loss(agent, agent_prime, batch, Args())
                a_loss = actor_loss(torch.FloatTensor(obs), current_option, logp, entropy, reward, done, torch.FloatTensor(next_obs), agent, agent_prime, Args())
                loss = a_loss + c_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            obs = next_obs
            total_steps += 1
            
        print(f"Episode {ep} reward: {ep_reward}")

if __name__ == "__main__":
    train_option_critic_with_demos()
