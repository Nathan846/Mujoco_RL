import mujoco
import mujoco_viewer
import numpy as np
import time
import gym
from gym import spaces
from contact_force_modelling import ContactForce
class MuJoCoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, model_path):
        super(MuJoCoEnv, self).__init__()
        self.integration_dt = 1.0
        self.damping = 1e-4
        self.gravity_compensation = True
        self.dt = 0.002
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
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)
        self.error_init()
        self.done = False
        self.add_glass_slab()
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

    def add_glass_slab(self, position=[0.2,0.3,0.1]):
        slab_mocap_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
        self.data.mocap_pos[slab_mocap_id-1] = np.array([1.0 , 0.0, 0.5])
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
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mjv_defaultFreeCamera(self.model, self.viewer.cam)
        mujoco.mj_resetData(self.model, self.data)
        self.done = False    
        self.data.qpos[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

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

    def _get_obs(self):
        return self.get_end_effector_position()

    def get_end_effector_position(self):
        return self.data.site_xpos[self.site_id].copy()

    def step(self, action):
        self.step_start = time.time()
        self.data.mocap_pos[self.mocap_id, : 3] = self.get_slab_midpoint()
        obs = self._get_obs()
        self.error_pos[:] = self.data.mocap_pos[self.mocap_id] - obs
        reward = -np.linalg.norm(self.error_pos)
        mujoco.mju_mat2Quat(self.site_quat, self.data.site_xmat[self.site_id])
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)
        if self.max_angvel > 0:
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > self.max_angvel:
                dq *= self.max_angvel / dq_abs_max

        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
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
        self.done = False
        return obs, reward, self.done, {}

    def render(self, mode="human"):
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


import time
env = MuJoCoEnv("universal_robots_ur5e/scene.xml")

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    time.sleep(0.01)
