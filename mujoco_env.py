"""
Env file for OA_env.py
"""
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np
import mujoco
import mujoco_viewer
import sys
import json
import time
from threading import Thread
from scipy.spatial.transform import Rotation as R
class MuJoCoEnv:
    def __init__(self, model_path):
        self.integration_dt = 1.0
        self.resolution = 0.05
        self.damping = 1e-4
        self.gravity_compensation = True
        self.dt = 0.00002
        self.contact_made = False
        self.welded = False 
        self.max_angvel = 1.0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
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
        init_quat = [1]
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1,0,0,0]
        self.data.qvel[8:11] = [0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.render()

    def render(self):
        if(self.welded):
            self.update_slab_to_match_eef()
        mujoco.mj_forward(self.model, self.data)
        try:
            pass
            # self.viewer.render()
        except:
            sys.exit(0)

    def update_slab_to_match_eef(self):
        eef_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        
        eef_pos = self.data.xpos[eef_body_id]
        eef_quat = self.data.xquat[eef_body_id]
        
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = eef_pos
        
        if not hasattr(self, 'eef_contact_quat'):
            self.eef_contact_quat = eef_quat.copy() 
            self.slab_contact_quat = self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7].copy()
        
        delta_quat = self.quaternion_multiply(eef_quat, self.quaternion_inverse(self.eef_contact_quat))
        
        new_slab_quat = self.quaternion_multiply(delta_quat, self.slab_contact_quat)
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = new_slab_quat

        mujoco.mj_forward(self.model, self.data)

    def quaternion_inverse(self, quat):
        """Compute the inverse of a quaternion (conjugate for unit quaternion)."""
        w, x, y, z = quat
        return np.array([w, -x, -y, -z])

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def simulate(self):
        while not self.contact_made:
            self.log_contact_forces()
            mujoco.mj_step(self.model, self.data)
            self.render()


    def close_application(self):
        print("Closing application...")
        self.close()
        self.window.close()

    def close(self):
        pass
        # self.viewer.close()
