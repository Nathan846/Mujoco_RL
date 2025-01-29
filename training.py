import json
import gym
from gym import spaces
import numpy as np
import mujoco
import mujoco_viewer
import json
import time


import numpy as np
def read_trajectory_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        init_data = data[0]
        init_pos = init_data.get("init_pos", [])
        init_quat = init_data.get("init_quat", [])
        print(f"Initial Position: {init_pos}")
        print(f"Initial Quaternion: {init_quat}")

        logs = []
        for i in range(1, len(data)):
            logs.extend(data[i])

        log_dicts = {}
        for log in logs:    
            timestamp = log.get("timestamp", None)
            joint_angles = log.get("joint_angles", [])
            contact = log.get("contact", {})
            contact_force = contact["forces"]["full_contact_force"]

            if contact_force == 0:
                continue

            g1 = contact['geom1']
            g2 = contact['geom2']
            if timestamp not in log_dicts:
                log_dicts[timestamp] = {"obs": [0] * 5, "action": joint_angles}

            contact_sums = log_dicts[timestamp]["obs"]
            if g1 == 31 and g2 == 33:
                contact_sums[0] += np.linalg.norm(contact_force)
            elif g1 == 31 and g2 == 36:
                contact_sums[1] += np.linalg.norm(contact_force)
            elif g1 == 31 and g2 == 39:
                contact_sums[2] += np.linalg.norm(contact_force)
            elif g1 == 31 and (g2 == 30 or g2 == 27):
                contact_sums[3] += 1  # Binary contact count
            else:
                contact_sums[4] += np.linalg.norm(contact_force)

        observations = [np.array(entry["obs"], dtype=np.float32) for entry in log_dicts.values()]
        actions = [np.array(entry["action"], dtype=np.float32) for entry in log_dicts.values()]

        return observations, actions

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON file.")
        return None, None
trajectory_file_path = "trajectories/gentle_place/place3.json"
obs, action = read_trajectory_file(trajectory_file_path)
class GymMuJoCoEnv(gym.Env):
    def __init__(self, model_path):
        super(GymMuJoCoEnv, self).__init__()

        self.dt = 0.00002
        self.contact_made = False
        self.welded = False 
        self.max_angvel = 1.0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.logs = []

        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]
        init_quat = [1, 0, 0, 0]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = init_quat
        self.data.qvel[8:11] = [0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

        n_joints = 7
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_joints,), dtype=np.float32)

        obs_dim = 5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def step(self, action):
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        print(self.get_eef_position())
        current_angles = self.data.qpos[:len(action)]
        delta_angles = action - current_angles

        self.data.ctrl[:len(action)] = delta_angles
        if(self.welded and not self.contact_made):
            self.update_slab_to_match_eef()
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self.contact_made

        self._log_state()

        return obs, reward, done, {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]
        init_quat = [1, 0, 0, 0]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = init_quat

        mujoco.mj_forward(self.model, self.data)
        self.contact_made = False
        self.logs = []

        return self._get_obs()
    def get_eef_position(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")

        body_position = data.xpos[body_id]

        body_orientation = self.data.xmat[body_id].reshape(3, 3)
        eef_relative_position = np.array([0, 0, 0.01]) 
        eef_world_position = body_position + body_orientation @ eef_relative_position

        return eef_world_position

    def render(self, mode='human'):
        
        mujoco.mj_forward(self.model, self.data)
        
        self.viewer.render()

    def close(self):
        self.viewer.close()

    def _get_obs(self):
        contact_sums = [0.0, 0.0, 0.0, 0.0, 0.0]

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_force = np.zeros(6)
            
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            full_contact_force = np.linalg.norm(contact_force[:3])
            if full_contact_force == 0:
                continue

            # Get geom IDs
            g1 = contact.geom1
            g2 = contact.geom2

            if g1 == 31 and g2 == 33:
                contact_sums[0] += full_contact_force
            elif g1 == 31 and g2 == 36:
                contact_sums[1] += full_contact_force
            elif g1 == 31 and g2 == 39:
                contact_sums[2] += full_contact_force
            elif g1 == 31 and (g2 == 30 or g2 == 27):
                contact_sums[3] += 1  # Binary contact count
            else:
                contact_sums[4] += full_contact_force

        # Return the summed contact forces as part of the observation
        return np.array(contact_sums, dtype=np.float32)
    def _compute_reward(self):
        reward = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            reward -= np.linalg.norm(contact_force)
        return reward

    def _log_state(self):
        joint_angles = self.data.qpos[:7].tolist()
        vel = self.data.qvel.tolist()
        slab_pos = self.data.qpos[7:10].tolist()
        slab_quat = self.data.qpos[10:14].tolist()

        log_entry = {
            "timestamp": time.time(),
            "joint_angles": joint_angles,
            "slab_position": slab_pos,
            "slab_orientation": slab_quat,
        }

        self.logs.append(log_entry)
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.utils import obs_as_tensor
import torch as th

# Load and process trajectory file
trajectory_file_path = "trajectories/gentle_place/place1.json"  # Replace with your file path
observations, actions = read_trajectory_file(trajectory_file_path)

env = GymMuJoCoEnv("universal_robots_ur5e/scene.xml")

model = PPO("MlpPolicy", env, verbose=1)

for obs, action in zip(observations, actions):
    
    # Expand observation to include batch dimension
    obs_tensor = obs_as_tensor(np.expand_dims(np.array(obs), axis=0), model.policy.device)
    # Expand action tensor to include batch dimension
    action_tensor = th.tensor(np.expand_dims(np.array(action), axis=0), dtype=th.float32, device=model.policy.device)
    
    # Forward pass through the policy to get action logits
    action_logits, value_estimate, log_prob = model.policy(obs_tensor)
    
    # Use action logits as the predicted action
    predicted_action = action_logits
    
    # Compute the loss (Mean Squared Error between predicted and actual actions)
    loss = th.nn.MSELoss()(predicted_action, action_tensor)
    
    # Optimize the policy
    model.policy.optimizer.zero_grad()
    loss.backward()
    model.policy.optimizer.step()
    env.render()
