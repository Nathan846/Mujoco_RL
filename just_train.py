import mujoco
import mujoco_viewer
import numpy as np
import time
from scipy.interpolate import interp1d
from contact_force_modelling import ContactForce

class MuJoCoDemoEnv:
    def __init__(self, model_path):
        self.dt = 0.01  # Simulation time step
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.joint_ids = [self.model.joint(name).id for name in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            print(f"Geom ID: {geom_id}, Name: {geom_name}")
    def contact(self):
        with open("contact_forces_train.txt", "a") as file:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if (contact.geom1 == 1 and 
                    contact.geom2 == 29):
                    force_contact_frame = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, i, force_contact_frame)
                    print(force_contact_frame)
                    normal_force = force_contact_frame[0]
                    tangential_force_1 = force_contact_frame[1]
                    tangential_force_2 = force_contact_frame[2]

                    file.write(f"Contact {i} between geom {contact.geom1} and geom {contact.geom2}:\n")
                    file.write(f"  Normal force: {normal_force}\n")
                    file.write(f"  Tangential force 1: {tangential_force_1}\n")
                    file.write(f"  Tangential force 2: {tangential_force_2}\n")
                    file.write("\n")
                    return

    def load_trajectory(self, file_path, total_points=200, decel_points=20):
        """Loads and interpolates joint angles from a file."""
        joint_angles = np.loadtxt(file_path)  # Load data from text file (15x6 array)
        original_points = joint_angles.shape[0]
        
        # Time points for interpolation
        time_original = np.linspace(0, 1, original_points)
        time_linear = np.linspace(0, 0.9, total_points - decel_points)
        time_decel = np.linspace(0.9, 1, decel_points)
        time_new = np.concatenate((time_linear, time_decel))

        interpolated_angles = np.zeros((total_points, joint_angles.shape[1]))
        
        for joint in range(joint_angles.shape[1]):
            # Linear for the initial, cubic for the decelerated end
            linear_interp = interp1d(time_original, joint_angles[:, joint], kind='linear')
            cubic_interp = interp1d(time_original, joint_angles[:, joint], kind='cubic')
            interpolated_angles[:total_points - decel_points, joint] = linear_interp(time_linear)
            interpolated_angles[total_points - decel_points:, joint] = cubic_interp(time_decel)

        self.trajectory = interpolated_angles

    def execute_trajectory(self):
        """Executes the interpolated trajectory step-by-step."""
        for joint_angles in self.trajectory:
            # Set the joint positions to the interpolated angles
            self.data.qpos[:len(joint_angles)] = joint_angles
            mujoco.mj_forward(self.model, self.data)  # Update the simulation state
            self.contact() 
            self.viewer.render()  # Render the environment
            time.sleep(self.dt)  # Step time interval to match the simulation

        # Hold the arm at the final position
        final_position = self.trajectory[-1]
        while True:
            self.data.qpos[:len(final_position)] = final_position
            mujoco.mj_forward(self.model, self.data)
            self.contact() 
            self.viewer.render()
            time.sleep(self.dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Initialize environment and execute demo
env = MuJoCoDemoEnv("universal_robots_ur5e/scene.xml")
env.load_trajectory("joint_anglefile.txt")  # Load and interpolate joint angles
env.execute_trajectory()  # Run the demonstration

env.close()  # Close the environment
