import numpy as np
import sys
from fk import *
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Define UR5e joint limits (in radians)
joint_limits = [
    (-np.pi, np.pi),  # Joint 1
    (-np.pi/2, np.pi/2),  # Joint 2
    (-np.pi, np.pi),  # Joint 3
    (-np.pi, np.pi),  # Joint 4
    (-np.pi, np.pi),  # Joint 5
    (-np.pi, np.pi)   # Joint 6
]

# Function to generate random joint angles within limits
def random_joint_angles():
    return np.array([np.random.uniform(low, high) for low, high in joint_limits])

# Function to convert a transformation matrix to a quaternion
def quaternion_from_matrix(matrix):
    rot = R.from_matrix(matrix[:3, :3])
    return rot.as_quat()

# Generate 100,000 samples
num_samples = 10000000
data = []

for _ in range(num_samples):
    q = random_joint_angles()  # Generate random joint angles
    T = forward(q)  # Compute forward kinematics
    pos = T[:3, 3]  # Extract position
    quat = quaternion_from_matrix(T)  # Extract quaternion
    data.append(np.hstack((q, pos, quat)))  # Store data

# Create DataFrame
columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'eef_x', 'eef_y', 'eef_z', 'eef_qx', 'eef_qy', 'eef_qz', 'eef_qw']
df = pd.DataFrame(data, columns=columns)

df.to_csv('fk.csv')