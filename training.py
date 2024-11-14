import mujoco
import mujoco_viewer
import numpy as np
import glfw

class MuJoCoMouseControlEnv:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.damping = 1e-4
        self.mouse_pressed = False
        self.previous_mouse_pos = (0, 0)
        self.integration_dt = 1.0
        self.model.opt.timestep = 0.00001
        self.model.opt.gravity[:] = [0, 0, -9.81]

        self.joint_ids = [self.model.joint(name).id for name in [
            'elbow_joint', 'shoulder_lift_joint', 'shoulder_pan_joint', 
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint','arm3']]
        
        self.max_angvel = 1.0
        glfw.set_mouse_button_callback(self.viewer.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.viewer.window, self._mouse_move_callback)
        self.error_init()
        
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

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                x, y = glfw.get_cursor_pos(window)
                self.mouse_pressed = True
                self.previous_mouse_pos = (x, y)
            elif action == glfw.RELEASE:
                self.mouse_pressed = False

    def _mouse_move_callback(self, window, xpos, ypos):
        self.mouse_x = xpos
        self.mouse_y = ypos
        if self.mouse_pressed:
            dx, dy = xpos - self.previous_mouse_pos[0], ypos - self.previous_mouse_pos[1]
            self._move_end_effector_with_mouse(dx, dy)
    
    def _move_end_effector_with_mouse(self, dx, dy):
        target_eef_pos = self.get_mouse_world_coordinates()
        if target_eef_pos is not None:
            target_joint_angles = self.inverse_kinematics(target_eef_pos)
            print('joint angels', target_joint_angles)
            self.data.qpos[self.joint_ids] = target_joint_angles
        if target_eef_pos is None:
            print('yo codes cooked')

    def inverse_kinematics(self, target_pos, target_quat=[0.2, 0.3, 0, 0]):
        obs = self._get_obs()
        self.error_pos[:] = target_pos - obs
        mujoco.mju_mat2Quat(self.site_quat, self.data.site_xmat[self.site_id])
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)
        
        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
        dq_desired = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)
        
        max_dq_step = 0.01
        dq_clipped = np.clip(dq_desired, -max_dq_step, max_dq_step)
        print('q del', dq_clipped)
        q = self.data.qpos.copy()
        print('q bf', q)
        q += dq_clipped
        print('q ac', q)
        return q

    def _get_obs(self):
        return self.get_end_effector_position()
    
    def get_end_effector_position(self):
        return self.data.site_xpos[self.site_id].copy()
    
    def get_mouse_world_coordinates(self):
        mouse_x, mouse_y = glfw.get_cursor_pos(self.viewer.window)
        width, height = glfw.get_window_size(self.viewer.window)
        
        relx = mouse_x / width
        rely = mouse_y / height

        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        flexid = np.array([-1], dtype=np.int32)
        skinid = np.array([-1], dtype=np.int32)

        aspectratio = width / height

        body_id = mujoco.mjv_select(
            self.model, self.data, self.viewer.vopt,
            aspectratio, relx, rely, self.viewer.scn,
            selpnt, geomid, flexid, skinid
        )
        return selpnt
    
    def run(self):
        while not glfw.window_should_close(self.viewer.window):
            self.viewer.render()
            glfw.poll_events()
            mujoco.mj_step(self.model, self.data)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        glfw.terminate()

# Usage
env = MuJoCoMouseControlEnv("universal_robots_ur5e/scene.xml")
try:
    env.run()
finally:
    env.close()
