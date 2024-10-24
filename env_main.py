import mujoco
import glfw
import numpy as np
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_error(target_quat, current_quat):
    """
    Compute the rotational error between two quaternions.
    Returns the error in axis-angle format (scaled by the angle).
    """
    # Compute the relative quaternion (target * inv(current))
    rel_quat = R.from_quat(target_quat) * R.from_quat(current_quat).inv()

    # Convert to axis-angle representation (magnitude of the rotation around the axis)
    axis_angle = rel_quat.as_rotvec()

    return axis_angle

class GaussNewtonIK:
    def __init__(self, model, data, step_size=0.01, tol=0.001, alpha=0.5):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = np.zeros((3, model.nv))  # Translational Jacobian
        self.jacr = np.zeros((3, model.nv))  # Rotational Jacobian

    def check_joint_limits(self, delta_q):
        penalty_scale = 10.0  # Increase to make the penalty stronger
        q = self.data.qpos[:6]
        for i in range(len(q)):
            lower_limit, upper_limit = self.model.jnt_range[i]
            margin = 0.1 * (upper_limit - lower_limit)  # 10% margin near the limit
            
            if q[i] < lower_limit + margin:
                delta_q[i] += penalty_scale * (lower_limit + margin - q[i])
                print(f"Joint {i} approaching lower limit. Applying penalty.")
            elif q[i] > upper_limit - margin:
                delta_q[i] -= penalty_scale * (q[i] - (upper_limit - margin))
                print(f"Joint {i} approaching upper limit. Applying penalty.")
        return q
    def calculate(self, goal_pos, goal_quat, body_id):
        """
        Calculate desired joint angles to reach both the target position and orientation.
        
        :param goal_pos: Target position [x, y, z]
        :param goal_quat: Target orientation as quaternion [x, y, z, w]
        :param init_q: Initial joint angles
        :param body_id: ID of the body to control (usually the end-effector)
        """
        mujoco.mj_forward(self.model, self.data)

        # Get the current pose of the body (end-effector)
        current_pos = self.data.body(body_id).xpos
        current_quat = self.data.body(body_id).xquat

        # Compute both position and rotation error
        pos_error = np.subtract(goal_pos, current_pos)
        

        rot_error = quaternion_error(goal_quat, current_quat)

        # Combine position and rotation errors into a 6D error vector
        error = np.hstack((pos_error, rot_error))

        max_iterations = 1000
        iterations = 0
        lambda_ = 1e-3  # Damping factor for regularization
        previous_error = np.inf
        errors = []
        while np.linalg.norm(error) >= self.tol and iterations < max_iterations:
            # Get the position and orientation of the point for which to compute the Jacobian
            point = self.data.body(body_id).xpos.copy()
            current_error = np.linalg.norm(error)
            # Compute the Jacobian for both position and orientation
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, point, body_id)

            # Combine translational and rotational Jacobians into a single 6xN Jacobian
            full_jacobian = np.vstack((self.jacp, self.jacr))
            U, S, Vt = np.linalg.svd(full_jacobian, full_matrices=False)
            S_inv = np.diag([1/s if s > 1e-5 else 0 for s in S])  # Handle small singular values
            j_inv = Vt.T @ S_inv @ U.T
            delta_q = j_inv @ error
            print(f"{iterations} it delta q val {delta_q}")

            # Update joint positions with step size
            self.data.qpos[:6] += delta_q
            self.data.qpos = self.check_joint_limits(self.data.qpos[:6])

            # Compute forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Recompute error (both position and rotation)
            current_pos = self.data.body(body_id).xpos
            current_quat = self.data.body(body_id).xquat

            # Recompute the errors
            pos_error = np.subtract(goal_pos, current_pos)
            print(f"Goal quaternion: {goal_quat}")
            print(f"Current quaternion: {current_quat}")
            rot_error = quaternion_error(goal_quat, current_quat)

            error = np.hstack((pos_error, rot_error))

            if np.linalg.norm(error) < 1e-5:
                break

            iterations += 1


        print(f"Final error after {iterations} iterations: {np.linalg.norm(error)}")
        return self.data.qpos[:6]
class MuJoCoSimulator:
    def __init__(self, model_path, window_width=800, window_height=600):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.window_width = window_width
        self.window_height = window_height
        self.gripper_state = False
        self.ik_solver = GaussNewtonIK(self.model, self.data)
        self.init_q = np.zeros(6)
        self.is_rotating = False
        self.is_panning = False
        self.is_zooming = False
        self.last_x, self.last_y = 0.0, 0.0
        self.cam_azimuth_speed = 0.2
        self.cam_elevation_speed = 0.2
        self.zoom_speed = 0.05
        self.pan_speed = 0.01

        self._init_glfw()
        self._init_scene()
    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            ee_position = self.get_end_effector_position()
            if key == glfw.KEY_LEFT:
                ee_position[1] -= 0.01
            elif key == glfw.KEY_RIGHT:
                ee_position[1] += 0.01
            elif key == glfw.KEY_UP:
                ee_position[0] += 0.01
            elif key == glfw.KEY_DOWN:
                ee_position[0] -= 0.01
            self.move_end_effector(ee_position, self.get_end_effector_position())

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.is_rotating = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.is_panning = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.is_zooming = action == glfw.PRESS

    def cursor_position_callback(self, window, xpos, ypos):
        dx, dy = xpos - self.last_x, ypos - self.last_y
        self.last_x, self.last_y = xpos, ypos

        if self.is_rotating:
            self.cam.azimuth += dx * self.cam_azimuth_speed
            self.cam.elevation = np.clip(self.cam.elevation - dy * self.cam_elevation_speed, -89.9, 89.9)
        elif self.is_panning:
            self.cam.lookat[0] -= dx * self.pan_speed
            self.cam.lookat[1] += dy * self.pan_speed
        elif self.is_zooming:
            self.cam.distance = max(0.1, self.cam.distance - dy * self.zoom_speed)

    def scroll_callback(self, window, xoffset, yoffset):
        self.cam.distance = max(0.1, self.cam.distance - yoffset * self.zoom_speed)

    def _init_glfw(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        self.window = glfw.create_window(self.window_width, self.window_height, "MuJoCo Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        # glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_position_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)

    def _init_scene(self):
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.cam.azimuth = 0
        self.cam.elevation = -30
        self.cam.distance = 5  # Adjust if necessary
        self.cam.lookat[:] = np.array([0, 0, 0])  # Adjust based on your robot's position
    def get_end_effector_position(self):
        wrist_3_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "vacuum_gripper")
        return self.data.site_xpos[wrist_3_site_id].copy()

    def move_end_effector(self, goal, quat):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vacuum_gripper")
        final_qpos = self.ik_solver.calculate(goal, quat, body_id)
        self.data.qpos[:6] = final_qpos

    # def new_target_position(self, current_pos, target_pos, epsilon=0.01, convergence_threshold=1e-5):
    #     new_target_pos = current_pos.copy()

    #     for i in range(3):
    #         displacement = target_pos[i] - current_pos[i]
    #         if abs(displacement) < convergence_threshold:
    #             new_target_pos[i] = target_pos[i]
    #         else:
    #             adaptive_epsilon = epsilon if displacement > epsilon else displacement
    #             step = np.clip(displacement, -adaptive_epsilon, adaptive_epsilon)
    #             new_target_pos[i] += step
    #     return new_target_pos

    def render(self):
        width, height = glfw.get_framebuffer_size(self.window)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self.scene, self.context)

    def update(self):
        mujoco.mj_step(self.model, self.data)
        self.render()

    def run(self):
        goal_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        target_pos = np.array([0.8, 1, 0.6])
        while not glfw.window_should_close(self.window):
            self.move_end_effector(target_pos, goal_orientation)
            self.update()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(0.01)

        glfw.terminate()



if __name__ == "__main__":
    simulator = MuJoCoSimulator("ur10e/ur10e.xml")
    simulator.run()
