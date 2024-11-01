import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

class MujocoInverseKinematics:
    def __init__(self, model, data, step_size=0.1, tol=0.001, alpha=0.5):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha

    def quaternion_error(self, target_quat, current_quat):
        rel_quat = R.from_quat(target_quat) * R.from_quat(current_quat).inv()
        axis_angle = rel_quat.as_rotvec()
        return axis_angle

    def calculate(self, goal_pos, goal_quat, body_id, max_iterations=1000):
        mujoco.mj_forward(self.model, self.data)
        current_pos = self.data.body(body_id).xpos
        current_quat = self.data.body(body_id).xquat
        pos_error = np.subtract(goal_pos, current_pos)
        rot_error = self.quaternion_error(goal_quat, current_quat)
        error = np.hstack((pos_error, rot_error))
        iterations = 0
        
        
        while np.linalg.norm(error) >= self.tol and iterations < max_iterations:
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.body(body_id).xpos
            current_quat = self.data.body(body_id).xquat
            pos_error = np.subtract(goal_pos, current_pos)
            rot_error = self.quaternion_error(goal_quat, current_quat)
            error = np.hstack((pos_error, rot_error))
            iterations += 1

        print(f"Final error after {iterations} iterations: {np.linalg.norm(error)}")
        return self.data.qpos[:6]