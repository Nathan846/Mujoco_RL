import mujoco
import mujoco_viewer
import numpy as np
import time
import gym
import random
from gym import spaces
from contact_force_modelling import ContactForce
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from scipy.interpolate import interp1d

class MuJoCoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, model_path, training = False):
        super(MuJoCoEnv, self).__init__()
        self.integration_dt = 1.0
        self.damping = 1e-4
        self.training = training
        self.gravity_compensation = True
        self.dt = 0.01
        self.max_angvel = 1.0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.key_id = self.model.key("home").id
        self.mocap_id = self.model.body("slab_mocap").mocapid[0]
        self.grav_compensation()
        self.control_joints()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(6,), dtype=np.float32)        
        self.error_init()
        self.done = False
        self.add_glass_slab()
        self.dq = np.zeros(6)
        self.vacuum_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "adhesion_gripper")
        self.slab_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "slab")
        self.contact_force = ContactForce(self.model, self.data,self.slab_geom_id,self.vacuum_geom_id)
    def _get_info(self) -> dict:
        eef_position, _ = self._controller.get_eef_position()
        target_position = self._target.get_mocap_pose(self._physics)[0:3]
        distance_to_target = np.linalg.norm(eef_position - target_position)
        
        return {
            "eef_position": eef_position,
            "target_position": target_position,
            "distance_to_target": distance_to_target,
        }

    def add_glass_slab(self, position=[0,0,0]):
        slab_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
        # self.data.mocap_pos[slab_mocap_id-1] = np.array([1.0 , 0.0, 0.5])
        mujoco.mj_forward(self.model, self.data)

    def get_slab_midpoint(self):
        self.slab_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
        slab_midpt = self.data.xpos[self.slab_body_id]
        return slab_midpt
    def grav_compensation(self):
        body_names = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
        body_ids = [self.model.body(name).id for name in body_names]
        if self.gravity_compensation:
            for body_id in body_ids:
                self.model.body_gravcomp[body_id] = 1.0 if "wrist_3_link" in body_names else 0
    def control_joints(self):
        joint_names = ['elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        ac_ids =  ['adhere', 'elbow', 'shoulder_lift', 'shoulder_pan', 'wrist_1', 'wrist_2', 'wrist_3']
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in ac_ids])

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        mujoco.mj_resetData(self.model, self.data)
        self.done = False    
        self.data.qpos = [4.84, -0.314, -0.251, -1.45, -1.7608, 0]
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def get_trajectory(self, joint_angles, total_points=200, decel_points=20):
        """
        Interpolates joint angles to a larger number of time points with deceleration towards the end.
        
        Parameters:
        - joint_angles: numpy array of shape (15, 6), original time points.
        - total_points: int, total interpolated points desired.
        - decel_points: int, the number of points at the end with decelerated interpolation.
        
        Returns:
        - interpolated_angles: numpy array of shape (total_points, 6), with interpolated values.
        """
        original_points = joint_angles.shape[0]
        
        # Define time ranges for interpolation
        time_original = np.linspace(0, 1, original_points)
        time_linear = np.linspace(0, 0.9, total_points - decel_points)
        time_decel = np.linspace(0.9, 1, decel_points)
        time_new = np.concatenate((time_linear, time_decel))

        interpolated_angles = np.zeros((total_points, joint_angles.shape[1]))

        for joint in range(joint_angles.shape[1]):
            linear_interp = interp1d(time_original, joint_angles[:, joint], kind='linear')
            cubic_interp = interp1d(time_original, joint_angles[:, joint], kind='cubic')
            
            interpolated_angles[:total_points - decel_points, joint] = linear_interp(time_linear)
            interpolated_angles[total_points - decel_points:, joint] = cubic_interp(time_decel)

        return interpolated_angles

    def training_step(self):
        data = np.loadtxdt("joint_anglefile.txt")

        self.get_trajectory(data)
    
    def error_init(self):
        self.jac = np.zeros((6, self.model.nv))
        self.diag = self.damping * np.eye(6)
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.site_id = self.model.site("attachment_site").id
    def calculate_reward(self):
        distance_to_target = np.linalg.norm(self.error_pos[:3])*500
        
        reward = -distance_to_target
        ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        
        for contact in self.data.contact[:self.data.ncon]:
            if (contact.geom1 == 0) or \
               (contact.geom2 == ground_geom_id):
                reward -= 5000
                print("Penalty applied for ground contact!")
                continue
            if(contact.geom1 in [23,30] and contact.geom2 in [23,30]):
                continue
            if(contact.geom1 not in [1,30] or contact.geom2 not in [1,30]):
                print(contact)
                reward -= 1000
                print("touching ")
            if(contact.geom1 in [1,30] and contact.geom2 in [1,30]):
                reward += 1000
                print("this is what we want")
                

        return reward
    def _get_obs(self):
        return self.get_end_effector_position()

    def get_end_effector_position(self):
        return self.data.site_xpos[self.site_id].copy()

    def step(self, action):
        # if(self.training == True):
        #     self.training_step()
        self.step_start = time.time()
        self.data.mocap_pos[self.mocap_id, : 3] = self.get_slab_midpoint()
        obs = self._get_obs()
        self.error_pos[:] = self.data.mocap_pos[self.mocap_id] - obs
        reward = self.calculate_reward()
        mujoco.mju_mat2Quat(self.site_quat, self.data.site_xmat[self.site_id])
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
        self.dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)
        
        # self.dq += action
        if self.max_angvel > 0:
            dq_abs_max = np.abs(self.dq).max()
            if dq_abs_max > self.max_angvel:
                self.dq *= self.max_angvel / dq_abs_max
        print(reward)
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, self.dq, self.integration_dt)
        np.clip(q, *self.model.jnt_range.T, out=q)
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
        mujoco.mj_step(self.model, self.data)
        self.render()
        time_until_next_step = self.dt - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.contact_force.contact_pts()
        self.contact_force.gripper_force()
        self.done = np.linalg.norm(self.error_pos) < 0.01
        return obs, reward, self.done, {}

    def render(self, mode="human"):
        self.viewer.render()

    def close(self):
        print(self.error)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
import time
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = MuJoCoEnv("universal_robots_ur5e/scene.xml")
# done = True
# while done:
#     action = random.random()
#     env.step(action)
env = Monitor(env)

model = PPO("MlpPolicy", env, verbose=1, n_steps=250)

num_cycles = 1000
timesteps_per_cycle = 1000

for cycle in range(num_cycles):
    print(f"Starting training cycle {cycle + 1} of {num_cycles}")
    
    model.learn(total_timesteps=timesteps_per_cycle)
    print(f"Training cycle {cycle + 1} complete")
    print("ok ac comp")

    model.save(f"ppo_ur5e_cycle_{cycle + 1}")

    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    # print(f"Cycle {cycle + 1} - Mean reward: {mean_reward} +/- {std_reward}")

    obs = env.reset()


print("Training complete for all cycles.")
print(f"Demonstrating model performance for cycle {cycle + 1}")
done = True
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
    if done:
        obs = env.reset()
        
print(f"Cycle {cycle + 1} completed and saved. Moving to the next cycle.\n")