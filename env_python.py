import gym
import mujoco
import numpy as np
import os
import mujoco_viewer
class UR5Env(gym.Env):
    def __init__(self):
        super(UR5Env, self).__init__()
        xml_path = 'ur10e/ur10e.xml'
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        num_actuators = self.model.nu
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(num_actuators,), dtype=np.float32
        )

        observation_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32
        )

        self.viewer = None

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        end_effector_pos = self.data.body('wrist_3_link').xpos
        target_pos = self.data.body('glass_frame').xpos
        reward = -np.linalg.norm(end_effector_pos - target_pos)

        done = False

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        qpos = self.data.qpos.ravel()
        qvel = self.data.qvel.ravel()
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        if mode == 'human':
            self.viewer.render()

        elif mode == 'rgb_array':
            width, height = 640, 480
            img = self.viewer.read_pixels(width, height, depth=False)
            return np.flipud(img)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
