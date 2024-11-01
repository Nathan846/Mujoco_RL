import mujoco
import mujoco_viewer
import numpy as np

# Load the MuJoCo model for UR10e
model = mujoco.MjModel.from_xml_path('ur10e/ur10e.xml')  # Update with your UR10e model path
data = mujoco.MjData(model)

# Initialize the viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

# Target end-effector position (x, y, z in world coordinates)
target_position = np.array([1.5, 1.0, 2.5])  # Set your desired coordinates here

# IK parameters
tolerance = 0.01  # Position tolerance in meters
learning_rate = 0.05  # Step size for joint adjustments

# Get the site ID for the end effector
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')  # Update 'ee_site' to match your model's site name

def move_end_effector_to_target_while(target_position, model, data, viewer, tolerance=0.01, learning_rate=0.05):
    while True:
        # Update viewer
        viewer.render()

        # Calculate forward kinematics to get current end-effector position
        mujoco.mj_forward(model, data)
        ee_position = data.site_xpos[ee_site_id]  # Access end-effector position by site ID
        error = target_position - ee_position
        error_norm = np.linalg.norm(error)
        print(error_norm)
        # Check if the end effector is close enough to the target position
        if error_norm < tolerance:
            print("Target position reached!")
            break

        # Calculate Jacobian for end effector
        jacp = np.zeros((3, model.nv))  # Positional Jacobian (3xDOF)
        mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)

        # Calculate the required change in joint angles
        dq = learning_rate * np.dot(jacp.T, error)  # Jacobian transpose method for simplicity
        data.qpos[:model.nq] += dq  # Apply joint angle changes

        # Respect joint limits
        np.clip(data.qpos, model.jnt_range[:, 0], model.jnt_range[:, 1], out=data.qpos)

        # Step the simulation
        mujoco.mj_step(model, data)

    # Close the viewer after reaching the target
    viewer.close()

# Run the function with live rendering
move_end_effector_to_target_while(target_position, model, data, viewer, tolerance, learning_rate)
