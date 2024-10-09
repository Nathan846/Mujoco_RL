import gym
import mujoco
import numpy as np
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import OperationalSpaceController
import os
from dm_control import mjcf

# Now you can initialize physics
import numpy as np

def calculate_target_pose_for_OSC():
    desired_position = np.array([0.5, 0.0, 0.5])  # x, y, z coordinates
    desired_orientation = np.array([1, 0, 0, 0])  # qx, qy, qz, qw (no rotation)
    target_pose = np.concatenate([desired_position, desired_orientation])
    return target_pose

class SolarPanelEnv(gym.Env):
    def __init__(self):
        super(SolarPanelEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path('solar_gym.xml')
        self.data = mujoco.MjData(self.model)

        # Load UR5e arm and attach to the arena
        self._arm = Arm(
            xml_path=os.path.join(
                os.path.dirname(__file__),
                'ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )
        self._arena = StandardArena()
        self._arena.attach(self._arm.mjcf_model, pos=[0, -0.5, 0.2], quat=[0.7071068, 0, 0, -0.7071068])

        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        self._controller = OperationalSpaceController(
            physics=self._physics,
            joints=self._arm.joints,
            eef_site=self._arm.eef_site,
            min_effort=-150.0,
            max_effort=150.0,
            kp=200,
            ko=200,
            kv=50,
            vmax_xyz=1.0,
            vmax_abg=2.0,
        )

        # Set up gym observation and action spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Create viewer components (MuJoCo native)
        self.viewer_initialized = False
        self.camera = mujoco.MjvCamera()
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.option = mujoco.MjvOption()
        self.context = None

    def _init_viewer(self):
        # Initialize the context and make sure rendering is set up
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Set camera defaults
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.distance = 2.0
        self.camera.azimuth = 90.0
        self.camera.elevation = -45.0

        self.viewer_initialized = True

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        target_pose = calculate_target_pose_for_OSC()  
        self._controller.run(target_pose)

        mujoco.mj_step(self.model, self.data)

        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector')
        solar_panel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'solar_panel')
        reward = -np.linalg.norm(self.data.xpos[end_effector_id] - self.data.xpos[solar_panel_id])

        done = False  # Custom logic for done condition

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector')
        solar_panel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'solar_panel')
        end_effector_pos = self.data.xpos[end_effector_id]
        solar_panel_pos = self.data.xpos[solar_panel_id]
        self._physics.step()
        return np.concatenate([end_effector_pos, solar_panel_pos])

    def render(self, mode='human'):
        if not self.viewer_initialized:
            self._init_viewer()

        # Create a new buffer for rendering
        width, height = 800, 600
        viewport = mujoco.MjrRect(0, 0, width, height)

        # Update the scene
        mujoco.mjv_updateScene(self.model, self.data, self.option, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

        # Render the scene into the OpenGL context
        mujoco.mjr_render(viewport, self.scene, self.context)

    def close(self):
        if self.viewer_initialized and self.context is not None:
            mujoco.mjr_freeContext(self.context)
            self.viewer_initialized = False
