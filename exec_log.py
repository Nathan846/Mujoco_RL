import json
import numpy as np
import mujoco
import mujoco_viewer
import time
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys
# ==================== Trajectory Processing Functions ==================== #
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def flatten_json_structure(data):
    initial_values = data[0] if isinstance(data[0], dict) else None
    structured_data = []
    for entry in data:
        if isinstance(entry, list):
            structured_data.extend(entry)
    return initial_values, structured_data

def moving_average(array, window):
    return np.convolve(array, np.ones(window) / window, mode='valid')

def compute_difference(original, smoothed):
    return np.mean(np.abs(np.array(original) - np.array(smoothed)))

def remove_idle_segments(data, idle_threshold=6):
    cleaned_data = []
    last_joint_angles = None
    idle_count = 0
    zero_joint_seen = False

    for entry in data:
        current_joint_angles = tuple(entry["joint_angles"])
        if all(angle == 0 for angle in current_joint_angles):
            if zero_joint_seen:
                continue
            zero_joint_seen = True
        if last_joint_angles is not None and current_joint_angles == last_joint_angles:
            idle_count += 1
        else:
            idle_count = 0
        if idle_count < idle_threshold:
            cleaned_data.append(entry)
        last_joint_angles = current_joint_angles
    return cleaned_data

def resample_trajectory(data, target_frames=3000, extra_last_fraction=0.1, extra_frame_multiplier=2):
    num_original_frames = len(data)
    if num_original_frames <= target_frames:
        return data
    normal_fraction = 1.0 - extra_last_fraction
    normal_frames = int(target_frames * normal_fraction)
    extra_frames = target_frames - normal_frames
    split_index = int(num_original_frames * normal_fraction)
    normal_part = data[:split_index]
    extra_part = data[split_index:]
    normal_indices = np.linspace(0, len(normal_part) - 1, normal_frames).astype(int)
    extra_indices = np.linspace(0, len(extra_part) - 1, extra_frames).astype(int)
    final_indices = np.concatenate((normal_indices, extra_indices + split_index))
    return [data[idx] for idx in final_indices]

def smooth_trajectory_data(initial_values, data, window_size=5, target_frames=3000):
    if len(data) <= window_size:
        return {"initial_values": initial_values, "data": []}, 0
    data = remove_idle_segments(data, idle_threshold=6)
    data = resample_trajectory(data, target_frames=target_frames)
    joint_angles = np.array([entry["joint_angles"] for entry in data])
    smoothed_joint_angles = np.apply_along_axis(moving_average, axis=0, arr=joint_angles, window=window_size)
    smoothing_effect = compute_difference(joint_angles[:len(smoothed_joint_angles)], smoothed_joint_angles)
    smoothed_data = []
    for i in range(len(smoothed_joint_angles)):
        smoothed_data.append({
            "timestamp": data[i]["timestamp"],
            "joint_angles": smoothed_joint_angles[i].tolist(),
        })
    return {"initial_values": initial_values, "data": smoothed_data}, smoothing_effect

def save_json(data, output_path):
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

def process_json(input_file, output_file, window_size=5, target_frames=3000):
    raw_data = load_json(input_file)
    initial_values, structured_data = flatten_json_structure(raw_data)
    smoothed_data, smoothing_effect = smooth_trajectory_data(initial_values, structured_data, window_size, target_frames)
    save_json(smoothed_data, output_file)
    print(len(smoothed_data["data"]))
    print(f"Processed JSON saved to: {output_file}")

# ==================== Trajectory Execution Functions ==================== #
class ReExecuteTrajectory:
    def __init__(self, model_path, trajectory_file, log_file):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.trajectory, self.init_state = self.load_trajectory(trajectory_file)
        self.logs = []
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.dt = 0.02
        self.model.opt.timestep = self.dt
        self.logs = []
        self.prev_joint_angles = np.zeros(7)
        self.final_contact = False
        self.gui_initialized = False
        self.app = None
        self.LOG_FILE = log_file
        self.window = None
        self.welded = False
        self.simulation_thread = None
        self.initial_slab_quat = np.array([1, 0, 0, 0])
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1,0,0,0]
        self.data.qvel[8:11] = [0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.render()
    def render(self):
        if(self.welded):
            self.update_slab_to_match_eef()
        mujoco.mj_forward(self.model, self.data)
        try:
            self.viewer.render()
        except:
            self.write_logs(self.LOG_FILE)
            print(f"saved data to {self.LOG_FILE}")
            sys.exit(0)
    def update_slab_to_match_eef(self):
        eef_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        
        # Get EEF position and quaternion
        eef_pos = self.data.xpos[eef_body_id]
        eef_quat = self.data.xquat[eef_body_id]
        
        # Set position to match EEF
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = eef_pos
        
        # Compute relative rotation from initial slab orientation to EEF orientation
        initial_slab_quat_inv = self.quaternion_inverse(self.initial_slab_quat)
        relative_quat = self.quaternion_multiply(eef_quat, initial_slab_quat_inv)
        
        # Apply only the relative rotation to the initial slab orientation
        new_slab_quat = self.quaternion_multiply(relative_quat, self.initial_slab_quat)
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = new_slab_quat

        mujoco.mj_forward(self.model, self.data)
    def quaternion_inverse(self, quat):
        """Compute the inverse of a quaternion (conjugate for unit quaternion)."""
        w, x, y, z = quat
        return np.array([w, -x, -y, -z])

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def load_trajectory(self, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data["data"], data["initial_values"]
    def log_contact_forces(self):
        log_data = []
        
        # Capture joint angles
        joint_angles = self.data.qpos[:7].tolist()  # Assuming 7-DOF arm
        
        # Compute delta DOJ
        delta_doj = [joint_angles[i] - self.prev_joint_angles[i] for i in range(len(joint_angles))]
        print(delta_doj)
        self.prev_joint_angles = joint_angles
        # Store previous joint angles for next step
        self.prev_joint_angles = joint_angles.copy()

        # Get End-Effector position and orientation
        eef_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        eef_position = self.data.xpos[eef_body_id].tolist()  # Global position (3,)
        eef_orientation = self.data.xquat[eef_body_id].tolist()  # Global quaternion (4,)

        # Get Slab Position and Orientation
        slab_position = self.data.qpos[7:10].tolist()
        slab_orientation = self.data.qpos[10:14].tolist()

        # Iterate through contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)

            log_entry = {
                "timestamp": time.time(),
                "joint_angles": joint_angles,
                "delta_doj": delta_doj,  # Newly added
                "slab_position": slab_position,
                "slab_orientation": slab_orientation,
                "end_effector": {
                    "position": eef_position,
                    "orientation": eef_orientation
                },
                "contact": {
                    "geom1": contact.geom1,
                    "geom2": contact.geom2,
                    "forces": {
                        "normal_force": contact_force[0],
                        "tangential_force_x": contact_force[1],
                        "tangential_force_y": contact_force[2],
                        "full_contact_force": contact_force.tolist()
                    }
                }
            }
            log_data.append(log_entry)

        # Append log entry to logs
        self.logs.append(log_data)
    def execute_trajectory(self):
        for state in self.trajectory:
            joint_angles = state["joint_angles"]
            for i in range(len(joint_angles)):
                self.data.qpos[i] = joint_angles[i]
            self.log_contact_forces()
            mujoco.mj_forward(self.model, self.data)
            self.render()
    def write_logs(self, filename="cf_log.json"):
        with open(filename, "w") as file:
            json.dump(self.logs, file, indent=4)
        print(f"Logs saved to {filename}")

    def close(self):
        self.write_logs(self.LOG_FILE)
        self.viewer.close()

import os
import json
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def get_json_files(directory):
    """Retrieve all JSON files from a directory."""
    try:
        return [f for f in os.listdir(directory) if f.endswith(".json")]
    except FileNotFoundError:
        print(f"Directory {directory} not found!")
        return []

def process_all_json_files(input_folder, output_folder, model_path, window_size=1, target_frames=100000):
    os.makedirs(output_folder, exist_ok=True)
    json_files = get_json_files(input_folder)
    if not json_files:
        print("No JSON files found in the directory!")
        return
    
    for filename in json_files:
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, f"processed_{filename}")
        log_file = os.path.join(output_folder, f"log_{filename}")
        
        print(f"\nProcessing: {input_file} -> {output_file}")
        process_json(input_file, output_file, window_size=window_size, target_frames=target_frames)
        executor = ReExecuteTrajectory(model_path, output_file, log_file)
        try:
            executor.execute_trajectory()
        except KeyboardInterrupt:
            print("Execution interrupted.")
            break
        finally:
            executor.close()
if __name__ == "__main__":
    input_folder = "traj_trace"
    output_folder = "trajectories"
    model_path = "universal_robots_ur5e/scene.xml"
    process_all_json_files(input_folder, output_folder, model_path, window_size=1, target_frames=100000)

