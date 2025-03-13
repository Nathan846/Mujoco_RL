import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mujoco
import mujoco_viewer

class MuJoCoEnv:
    def __init__(self, model_path):
        self.model_path = model_path
        self.integration_dt = 1.0
        self.damping = 1e-4
        self.gravity_compensation = True
        self.dt = 0.00002
        self.contact_made = False
        self.welded = False 
        self.max_angvel = 1.0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.logs = []
        self.final_contact = False
        self.gui_initialized = False
        self.app = None
        self.window = None
        self.simulation_thread = None
        self.initial_slab_quat = np.array([1, 0, 0, 0])
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1,0,0,0]
        self.data.qvel[8:11] = [0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.render()
        # self.print_all_geoms()
        self.logs = [{"init_pos": slab_pos, "init_quat": [1]}]
        

    def reset(self):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.logs = []
        self.final_contact = False
        self.gui_initialized = False
        self.app = None
        self.window = None
        self.simulation_thread = None
        self.initial_slab_quat = np.array([1, 0, 0, 0])
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1,0,0,0]
        self.data.qvel[8:11] = [0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.render()

        return self.get_state()
    def check_done(self):
        """
        Check if, at the current timestep, there are contact forces between geom 31 and both geom 36 and geom 39.
        Returns True if both contacts are detected in the same timestep; otherwise, returns False.
        """
        contact_36 = False
        contact_39 = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if ((contact.geom1 == 31 and contact.geom2 == 36) or (contact.geom2 == 31 and contact.geom1 == 36)):
                contact_36 = True
            if ((contact.geom1 == 31 and contact.geom2 == 39) or (contact.geom2 == 31 and contact.geom1 == 39)):
                contact_39 = True
        return contact_36 and contact_39

    def compute_low_level_reward(self, state, action, next_state):
        env = self
        if not env.welded:
            # Phase 1: Bring end effector close to slab.
            # Get the end effector (eef) position and the slab position.
            eef_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
            slab_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
            eef_pos = env.data.xpos[eef_body_id]
            slab_pos = env.data.xpos[slab_body_id]
            distance = np.linalg.norm(eef_pos - slab_pos)
            # Also penalize large actions (encouraging smooth motion)
            smooth_penalty = 0.1 * np.linalg.norm(action)
            reward = -distance - smooth_penalty
        else:
            stand_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "stand_geom")
            eef_geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "4boxes")
            
            contact_force = 0.0
            for i in range(env.data.ncon):
                contact = env.data.contact[i]
                if ((contact.geom1 == stand_geom_id and contact.geom2 == eef_geom_id) or 
                    (contact.geom1 == eef_geom_id and contact.geom2 == stand_geom_id)):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, i, force)
                    contact_force = np.linalg.norm(force)
                    break
            reward = -contact_force
            threshold = 5.0
            if contact_force < threshold:
                reward += 10.0
        return reward

    def step(self, action):
        self.data.qpos[:7] += action  
        mujoco.mj_step(self.model, self.data)
        self.render()
        next_state = self.get_state()
        reward = self.compute_low_level_reward(state, action, next_state)
        
        
        done = self.check_done()
        transition = (state, current_option, action, reward, next_state, termination_prob, done)
        info = {}
        return next_state, reward, done, info
    def render(self):
        if(self.welded and not self.contact_made):
            self.update_slab_to_match_eef()
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()
    def get_state(self):
        joint_angles = self.data.qpos[:7].tolist()
        slab_pos = self.data.qpos[7:10].tolist()
        slab_quat = self.data.qpos[10:14].tolist()
        normal_force = 0.0
        if self.data.ncon > 0:
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, 0, contact_force)
            normal_force = contact_force[0]
        state = np.array(joint_angles + slab_pos + slab_quat + [normal_force], dtype=np.float32)
        return state

def process_trajectory(trajectory_data):
    data_points = trajectory_data.get("data", [])
    states = []
    actions = []
    options = []

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
    
    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "options": np.array(options)
    }

def load_trajectories_from_folder(folder_path):
    trajectories = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    print(f"Found 72 trajectory files in {folder_path}")
    for file in json_files:
        with open(file, 'r') as f:
            try:
                traj_data = json.load(f)
                processed = process_trajectory(traj_data)
                trajectories.append(processed)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    return trajectories

class DemoDataset(Dataset):
    def __init__(self, trajectories):
        states_list = []
        actions_list = []
        options_list = []
        for traj in trajectories:
            states_list.append(traj["states"])
            actions_list.append(traj["actions"])
            options_list.append(traj["options"])
        print(states_list)
        valid_size = states_list[0].shape[1]  # Get the expected size along dimension 1

        filtered_states = [arr for arr in states_list if arr.shape[1] == valid_size]

        # Concatenate only the valid ones
        self.states = np.concatenate(filtered_states, axis=0)
        self.actions = np.concatenate(actions_list, axis=0)
        self.options = np.concatenate(options_list, axis=0)
        print("Total samples:", len(self.states))
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.options[idx]

class HighLevelAgent(nn.Module):
    def __init__(self, input_dim, num_options):
        super(HighLevelAgent, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_options)
        )
    
    def forward(self, x):
        return self.fc(x)

class OptionCriticLowLevelAgent(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(OptionCriticLowLevelAgent, self).__init__()
        self.action_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Actions scaled to [-1, 1]
        )
        # Termination network
        self.termination_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Outputs probability
        )
    
    def forward(self, x):
        action = self.action_net(x)
        termination_prob = self.termination_net(x)
        return action, termination_prob

folder_path = "trajectories"  # Adjust as needed
demo_trajectories = load_trajectories_from_folder(folder_path)
demo_dataset = DemoDataset(demo_trajectories)
dataloader = DataLoader(demo_dataset, batch_size=5000, shuffle=True)

state_dim = 15
action_dim = 7
num_options = 2
high_agent = HighLevelAgent(input_dim=state_dim, num_options=num_options)
low_agents = [OptionCriticLowLevelAgent(input_dim=state_dim, action_dim=action_dim) for _ in range(num_options)]

high_optimizer = optim.Adam(high_agent.parameters(), lr=5e-5)
low_optimizers = [optim.Adam(low_agents[i].parameters(), lr=5e-5) for i in range(num_options)]

mse_loss_fn = nn.MSELoss()
num_pretrain_epochs = 20

for epoch in range(num_pretrain_epochs):
    total_loss = 0.0
    for states, actions, options in dataloader:
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        
        for opt in range(num_options):
            idx = (torch.LongTensor(options) == opt).nonzero(as_tuple=True)[0]
            if idx.nelement() == 0:
                continue
            states_opt = states_tensor[idx]
            actions_opt = actions_tensor[idx]
            
            pred_actions, _ = low_agents[opt](states_opt)
            loss = mse_loss_fn(pred_actions, actions_opt)
            
            low_optimizers[opt].zero_grad()
            loss.backward()
            low_optimizers[opt].step()
            
            total_loss += loss.item()
    print(f"Pretrain Epoch {epoch+1}/{num_pretrain_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Note: For the high-level agent, since we don't have explicit option labels from demonstrations,
# we can choose to initialize it randomly and let it learn during the RL phase.
termination_threshold = 0.5  
num_rl_episodes = 100
replay_buffer = []
model_path = "universal_robots_ur5e/scene.xml"
env = MuJoCoEnv(model_path)
initial_state = env.reset()
import time
for episode in range(num_rl_episodes):
    state = env.reset()  # Assuming env is your MuJoCo environment instance
    done = False
    current_option = None
    episode_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if current_option is None:
            q_values = high_agent(state_tensor)
            current_option = torch.argmax(q_values, dim=1).item()
        
        low_agent = low_agents[current_option]
        action_tensor, termination_prob_tensor = low_agent(state_tensor)
        action = action_tensor.squeeze(0).detach().numpy()
        termination_prob = termination_prob_tensor.item()
        
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # Store transition in replay buffer (for later RL updates).
        transition = (state, current_option, action, reward, next_state, termination_prob, done)
        replay_buffer.append(transition)
        
        # Decide whether to terminate the current option.
        if termination_prob > termination_threshold:
            current_option = None  # Signal to choose a new option next time step.
        time.sleep(1)
        state = next_state
    
    print(f"RL Episode {episode+1}/{num_rl_episodes} complete, Reward: {episode_reward:.2f}")
