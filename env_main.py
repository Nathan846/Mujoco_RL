import mujoco
import glfw
import numpy as np
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from GaussIK import GaussNewtonIK
from mj_inv import MujocoInverseKinematics
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
        self.ik_solver.set_initial_q(method="zero")

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
    def check_joint_limits(self):
        """
        Check if the current joint positions (qpos) are within joint limits.
        
        :param model: MuJoCo model object
        :param data: MuJoCo data object
        :return: Boolean indicating whether all joints are within limits
        """
        model = self.model
        data = self.data
        for i in range(model.njnt):
            # Get the joint limits from the model
            lower_limit, upper_limit = model.jnt_range[i]
            print(f"Upper limit {upper_limit} and Lower limit is {lower_limit}")
            # Get the current joint position
            current_pos = data.qpos[i]
            
            # Check if current joint position is within limits
            if current_pos < lower_limit or current_pos > upper_limit:
                print(f"Joint {i} is out of bounds!")
                print(f"Joint {i} position: {current_pos}")
                print(f"Allowed range: [{lower_limit}, {upper_limit}]")
                return False
            else:
                print(f"Joint {i} is within limits: {current_pos}")
        
        # If all joints are within limits
        return True

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
        self.check_joint_limits()
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
