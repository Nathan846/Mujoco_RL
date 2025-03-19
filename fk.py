import numpy as np
from math import cos, sin, sqrt
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Quaternion

def quaternion_from_matrix(matrix):
    """Convert a 4x4 homogeneous transformation matrix to a quaternion."""
    rot = R.from_matrix(matrix[:3, :3])
    return rot.as_quat()

def rotation_matrix(angle, direction):
    """Create a rotation matrix from an angle and axis."""
    return R.from_rotvec(angle * np.array(direction)).as_matrix()

def quaternion_matrix(quaternion):
    """Convert a quaternion to a 4x4 transformation matrix."""
    rot = R.from_quat(quaternion).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    return mat

def euler_from_quaternion(quaternion):
    """Convert a quaternion to Euler angles (roll, pitch, yaw)."""
    return R.from_quat(quaternion).as_euler('xyz')

def matrix2ros(np_pose):
    ros_pose = Pose()
    ros_pose.position.x = np_pose[0, 3]
    ros_pose.position.y = np_pose[1, 3]
    ros_pose.position.z = np_pose[2, 3]
    np_q = quaternion_from_matrix(np_pose)
    ros_pose.orientation.x = np_q[0]
    ros_pose.orientation.y = np_q[1]
    ros_pose.orientation.z = np_q[2]
    ros_pose.orientation.w = np_q[3]
    return ros_pose

def euler2ros(ur_pose):
    ros_pose = Pose()
    ros_pose.position.x = ur_pose[0]
    ros_pose.position.y = ur_pose[1]
    ros_pose.position.z = ur_pose[2]
    angle = sqrt(ur_pose[3] ** 2 + ur_pose[4] ** 2 + ur_pose[5] ** 2)
    direction = [i / angle for i in ur_pose[3:6]]
    np_T = rotation_matrix(angle, direction)
    np_q = quaternion_from_matrix(np_T)
    ros_pose.orientation.x = np_q[0]
    ros_pose.orientation.y = np_q[1]
    ros_pose.orientation.z = np_q[2]
    ros_pose.orientation.w = np_q[3]
    return ros_pose

def ros2matrix(ros_pose):
    np_pose = quaternion_matrix([ros_pose.orientation.x, ros_pose.orientation.y, 
                                 ros_pose.orientation.z, ros_pose.orientation.w])
    np_pose[0][3] = ros_pose.position.x
    np_pose[1][3] = ros_pose.position.y
    np_pose[2][3] = ros_pose.position.z
    return np_pose

def ros2euler(ros_pose):
    quaternion = (
        ros_pose.orientation.x,
        ros_pose.orientation.y,
        ros_pose.orientation.z,
        ros_pose.orientation.w)
    euler = euler_from_quaternion(quaternion)
    return np.array([ros_pose.position.x, ros_pose.position.y, ros_pose.position.z, euler[0], euler[1], euler[2]])

def forward(q):
    s = [sin(q[i]) for i in range(6)]
    c = [cos(q[i]) for i in range(6)]
    q23 = q[1] + q[2]
    q234 = q[1] + q[2] + q[3]
    s23, c23 = sin(q23), cos(q23)
    s234, c234 = sin(q234), cos(q234)
    
    d1, a2, a3, d4, d5, d6 = 0.163, -0.425, -0.392, 0.127, 0.1, 0.1
    
    T = np.eye(4)
    T[0, 0] = c234 * c[0] * s[4] - c[4] * s[0]
    T[0, 1] = c[5] * (s[0] * s[4] + c234 * c[0] * c[4]) - s234 * c[0] * s[5]
    T[0, 2] = -s[5] * (s[0] * s[4] + c234 * c[0] * c[4]) - s234 * c[0] * c[5]
    T[0, 3] = d6 * c234 * c[0] * s[4] - a3 * c23 * c[0] - a2 * c[0] * c[1] - d6 * c[4] * s[0] - d5 * s234 * c[0] - d4 * s[0]
    T[1, 0] = c[0] * c[4] + c234 * s[0] * s[4]
    T[1, 1] = -c[5] * (c[0] * s[4] - c234 * c[4] * s[0]) - s234 * s[0] * s[5]
    T[1, 2] = s[5] * (c[0] * s[4] - c234 * c[4] * s[0]) - s234 * c[5] * s[0]
    T[1, 3] = d6 * (c[0] * c[4] + c234 * s[0] * s[4]) + d4 * c[0] - a3 * c23 * s[0] - a2 * c[1] * s[0] - d5 * s234 * s[0]
    T[2, 0] = -s234 * s[4]
    T[2, 1] = -c234 * s[5] - s234 * c[4] * c[5]
    T[2, 2] = s234 * c[4] * s[5] - c234 * c[5]
    T[2, 3] = d1 + a3 * s23 + a2 * s[1] - d5 * (c23 * c[3] - s23 * s[3]) - d6 * s[4] * (c23 * s[3] + s23 * c[3])
    return T
