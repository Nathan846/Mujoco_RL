import mujoco
import mujoco_viewer
import numpy as np
import time
import gym
from gym import spaces
from contact_force_modelling import ContactForce
from scipy.interpolate import interp1d

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
        self.vacuum_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "adhesion_gripper")
        self.slab_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "slab")
        self.contact_force = ContactForce(self.model, self.data,self.slab_geom_id,self.vacuum_geom_id)
        self.mocapped = False
    def _get_info(self) -> dict:
        eef_position, _ = self._controller.get_eef_position()
        target_position = self._target.get_mocap_pose(self._physics)[0:3]
        distance_to_target = np.linalg.norm(eef_position - target_position)
        
        return {
            "eef_position": eef_position,
            "target_position": target_position,
            "distance_to_target": distance_to_target,
        }
    def detach_mocap(self):
        current_pos = self.data.mocap_pos[self.model.body("slab_mocap").mocapid].copy()
        current_quat = self.data.mocap_quat[self.model.body("slab_mocap").mocapid].copy()
        self.data.mocap_pos[self.model.body("slab_mocap").mocapid] = current_pos
        self.data.mocap_quat[self.model.body("slab_mocap").mocapid] = current_quat
    def attach_mocap(self):
        self.mocapped = True
        end_effector_pos = self.data.xpos[self.model.body("4boxes").id]
        end_effector_quat = self.data.xquat[self.model.body("4boxes").id]
        self.data.mocap_pos[self.model.body("slab_mocap").mocapid] = end_effector_pos
        self.data.mocap_quat[self.model.body("slab_mocap").mocapid] = end_effector_quat
    def contact(self):
        with open("contact_forces_train.txt", "a") as file:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if (contact.geom1 == 31 and 
                    contact.geom2 == 34):
                    force_contact_frame = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force_contact_frame)
                    self.attach_mocap()
                    normal_force = force_contact_frame[0]
                    tangential_force_1 = force_contact_frame[1]
                    tangential_force_2 = force_contact_frame[2]

                    file.write(f"Contact {i} between geom {contact.geom1} and geom {contact.geom2}:\n")
                    file.write(f"  Normal force: {normal_force}\n")
                    file.write(f"  Tangential force 1: {tangential_force_1}\n")
                    file.write(f"  Tangential force 2: {tangential_force_2}\n")
                    file.write("\n")
                    return

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
        joint_names = [
            'arm3',                # Custom joint name defined in <body> "4boxes"
            'elbow_joint',         # Elbow joint in the forearm link
            'shoulder_lift_joint', # Shoulder lift joint in the upper arm link
            'shoulder_pan_joint',  # Shoulder pan joint in the shoulder link
            'wrist_1_joint',       # Wrist 1 joint in the wrist_1_link
            'wrist_2_joint',       # Wrist 2 joint in the wrist_2_link
            'wrist_3_joint'        # Wrist 3 joint in the wrist_3_link
        ]
        
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        
        actuator_names = [
            'shoulder_pan',    # Actuator for shoulder pan joint
            'shoulder_lift',   # Actuator for shoulder lift joint
            'elbow',           # Actuator for elbow joint
            'wrist_1',         # Actuator for wrist 1 joint
            'wrist_2',         # Actuator for wrist 2 joint
            'wrist_3',         # Actuator for wrist 3 joint
            'adhere_wrist'     # Actuator for the adhesion mechanism
        ]
        
        self.actuator_ids = np.array([self.model.actuator(name).id for name in actuator_names])

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
        if(self.mocapped):
            self.load_trajectory(file_path="move_angles.txt")
            print('two time cap')
            self.execute_trajectory()
            self.detach_mocap()
            self.mocapped = False
        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
        np.clip(q, *self.model.jnt_range.T, out=q)
        self.data.ctrl[self.actuator_ids] = q[self.dof_ids]
        mujoco.mj_step(self.model, self.data)
        self.render()
        time_until_next_step = self.dt - (time.time() - self.step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        self.done = np.linalg.norm(self.error_pos) < 0.01
        self.done = False
        return obs, reward, self.done, {}
    def load_trajectory(self, file_path, total_points=200, decel_points=20):
        joint_angles = np.loadtxt(file_path)  # Load data from text file (15x6 array)
        original_points = joint_angles.shape[0]
        
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

        self.trajectory = interpolated_angles

    def execute_trajectory(self):
        for joint_angles in self.trajectory:
            self.data.qpos[:len(joint_angles)] = joint_angles
            mujoco.mj_forward(self.model, self.data)  
            self.contact() 
            self.viewer.render() 
            time.sleep(self.dt)
        print('done traj thing')
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
env.load_trajectory("joint_anglefile.txt")
env.execute_trajectory()

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    time.sleep(0.01)
