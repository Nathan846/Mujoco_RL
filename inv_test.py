import mujoco
import numpy as np

# Set up your model and data
model = mujoco.MjModel.from_xml_path('ur10e/ur10e.xml')
data = mujoco.MjData(model)

# Configure the solver
model.opt.tolerance = 1e-5  # Tight tolerance
model.opt.iterations = 1000  # Increase iteration limit

# Target position and orientation
target_pos = np.array([0.5, 0.3, 0.2])
target_quat = np.array([0.707, 0, 0, 0.707])  # Example quaternion for 90 degree rotation

# Call MuJoCo's inverse kinematics solver
mujoco.mj_inverse(model, data)

# Now use forward kinematics to verify the result
mujoco.mj_forward(model, data)
def check_joint_limits(model, data):
    """
    Check if the current joint positions (qpos) are within joint limits.
    
    :param model: MuJoCo model object
    :param data: MuJoCo data object
    :return: Boolean indicating whether all joints are within limits
    """
    for i in range(model.njnt):
        # Get the joint limits from the model
        lower_limit, upper_limit = model.jnt_range[i]
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
check_joint_limits(model, data)
# Retrieve the computed position and orientation of the end-effector
computed_pos = data.body("wrist_3_link").xpos
computed_quat = data.body("wrist_3_link").xquat

# Print the results
print(f"Target position: {target_pos}, Computed position: {computed_pos}")
print(f"Target orientation: {target_quat}, Computed orientation: {computed_quat}")

# Calculate and display errors
position_error = np.linalg.norm(target_pos - computed_pos)
orientation_error = np.linalg.norm(target_quat - computed_quat)

print(f"Position error: {position_error}")
print(f"Orientation error: {orientation_error}")
