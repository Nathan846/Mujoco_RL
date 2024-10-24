
import numpy as np
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
class GaussNewtonIK:
    def __init__(self, model, data, step_size=0.01, tol=0.001, alpha=0.5):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.damping = 1e-3
        self.alpha = alpha
        self.jacp = np.zeros((3, model.nv)) 
        self.jacr = np.zeros((3, model.nv))
    def set_initial_q(self, method="random", predefined_q=None):
        q_init = np.zeros(self.model.nv)
        
        if method == "random":
            for i in range(self.model.nv):
                lower_limit, upper_limit = self.model.jnt_range[i]
                q_init[i] = np.random.uniform(lower_limit, upper_limit)
            print(f"Initialized joint values randomly: {q_init}")
        
        elif method == "zero":
            for i in range(self.model.nv):
                lower_limit, upper_limit = self.model.jnt_range[i]
                if lower_limit <= 0 <= upper_limit:
                    q_init[i] = 0.0
                else:
                    q_init[i] = (lower_limit + upper_limit) / 2.0 
            print(f"Initialized joint values to zero (or midpoint): {q_init}")
        
        elif method == "predefined" and predefined_q is not None:
            if len(predefined_q) == self.model.nv:
                q_init = predefined_q
                print(f"Initialized joint values to predefined values: {q_init}")
            else:
                raise ValueError(f"Predefined q must be of length {self.model.nv}")
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        self.data.qpos[:self.model.nv] = q_init
        return q_init
    def quaternion_error(self,target_quat, current_quat):
        rel_quat = R.from_quat(target_quat) * R.from_quat(current_quat).inv()

        axis_angle = rel_quat.as_rotvec()

        return axis_angle

    def check_joint_limits(self, q):
        for i in range(len(q)):
            lower_limit, upper_limit = self.model.jnt_range[i]
            q[i] = max(lower_limit, min(q[i], upper_limit))
        return q
    def calculate(self, goal_pos, goal_quat, body_id, max_iterations=1000):
        mujoco.mj_forward(self.model, self.data)
        
        current_pos = self.data.body(body_id).xpos
        current_quat = self.data.body(body_id).xquat
        
        pos_error = np.subtract(goal_pos, current_pos)
        rot_error = self.quaternion_error(goal_quat, current_quat)
        
        error = np.hstack((pos_error, rot_error))

        iterations = 0
        while np.linalg.norm(error) >= self.tol and iterations < max_iterations:
            
            point = self.data.body(body_id).xpos.copy()

            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, point, body_id)

            full_jacobian = np.vstack((self.jacp, self.jacr))
            n = full_jacobian.shape[1]
            I = np.identity(n)
            product = full_jacobian.T @ full_jacobian + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                print("Singular matrix detected, using pseudo-inverse")
                j_inv = np.linalg.pinv(product) @ full_jacobian.T
            else:
                j_inv = np.linalg.inv(product) @ full_jacobian.T
            
            delta_q = j_inv @ error
            
            self.data.qpos[:n] += self.step_size * delta_q

            self.data.qpos[:n] = self.check_joint_limits(self.data.qpos[:n])
            
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.body(body_id).xpos
            current_quat = self.data.body(body_id).xquat
            
            pos_error = np.subtract(goal_pos, current_pos)
            rot_error = self.quaternion_error(goal_quat, current_quat)
            
            error = np.hstack((pos_error, rot_error))

            iterations += 1

        print(f"Final error after {iterations} iterations: {np.linalg.norm(error)}")
        return self.data.qpos[:n]
