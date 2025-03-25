"""
Smoothens, and rebalances the original trajectory files for training
"""
import json
import numpy as np
import time
from scipy.interpolate import interp1d

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def flatten_json_structure(data):
    """Extracts initial values and flattens the JSON structure into a list of trajectory data."""
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
    """Removes idle segments and retains only one zero-joint-angle entry."""
    cleaned_data = []
    last_joint_angles = None
    idle_count = 0
    zero_joint_seen = False  # Track if a zero-joint entry has been added

    for entry in data:
        current_joint_angles = tuple(entry["joint_angles"])

        # Check if all joint angles are zero
        if all(angle == 0 for angle in current_joint_angles):
            if zero_joint_seen:
                continue  # Skip duplicate zero joint entries
            zero_joint_seen = True  # Mark that we've added one

        # Check for idle movements
        if last_joint_angles is not None and current_joint_angles == last_joint_angles:
            idle_count += 1
        else:
            idle_count = 0  # Reset idle count if movement is detected

        if idle_count < idle_threshold:
            cleaned_data.append(entry)

        last_joint_angles = current_joint_angles

    return cleaned_data
def resample_trajectory(data, target_frames=3000, extra_last_fraction=0.1, extra_frame_multiplier=2):
    """Resamples trajectory with extra emphasis on the last fraction of the trajectory."""
    
    num_original_frames = len(data)
    if num_original_frames <= target_frames:
        return data  # No need to resample if already within the limit

    # Determine normal and extra frame counts
    normal_fraction = 1.0 - extra_last_fraction
    normal_frames = int(target_frames * normal_fraction)
    extra_frames = target_frames - normal_frames

    split_index = int(num_original_frames * normal_fraction)  # 90% split
    normal_part = data[:split_index]
    extra_part = data[split_index:]

    # Ensure resampling does not exceed available data
    normal_frames = min(len(normal_part), normal_frames)
    extra_frames = min(len(extra_part) * extra_frame_multiplier, extra_frames)

    # Resample both sections separately
    normal_indices = np.linspace(0, len(normal_part) - 1, normal_frames).astype(int)
    extra_indices = np.linspace(0, len(extra_part) - 1, extra_frames).astype(int)

    final_indices = np.concatenate((normal_indices, extra_indices + split_index))  # Adjust extra indices

    print(f"Original Frames: {num_original_frames}, Target Frames: {target_frames}")
    print(f"Frames Allocated: {normal_frames} (First 90%) + {extra_frames} (Last 10%)")

    def interpolate_field(field, nested=False):
        """Handles interpolation of both normal and nested fields."""
        try:
            if nested:
                original_values = np.array([entry[field[0]][field[1]][field[2]] for entry in data])
            else:
                original_values = np.array([entry[field] for entry in data])

            # Create interpolation function and interpolate
            x_original = np.linspace(0, num_original_frames - 1, num_original_frames)
            x_target = np.linspace(0, num_original_frames - 1, target_frames)

            interp_func = interp1d(x_original, original_values, axis=0, kind='linear', fill_value="extrapolate")
            return interp_func(x_target)
        except KeyError as e:
            print(f"Error: Missing field {field} in trajectory data.")
            raise e

    # Create interpolated dataset
    interpolated_data = []
    for i in range(target_frames):  
        idx = final_indices[i]
        entry = {
            "timestamp": float(data[idx]["timestamp"]),
            "joint_angles": interpolate_field("joint_angles")[i].tolist(),
            "slab_position": interpolate_field("slab_position")[i].tolist(),
            "slab_orientation": interpolate_field("slab_orientation")[i].tolist(),
            "contact": {
                "geom1": data[idx]["contact"]["geom1"],
                "geom2": data[idx]["contact"]["geom2"],
                "forces": {
                    "normal_force": 0.0,
                    "tangential_force_x": 0.0,
                    "tangential_force_y": 0.0,
                    "full_contact_force": interpolate_field(["contact", "forces", "full_contact_force"], nested=True)[i].tolist()
                }
            }
        }
        interpolated_data.append(entry)

    return interpolated_data

def smooth_trajectory_data(initial_values, data, window_size=5, target_frames=3000):
    """Smooths trajectory, removes idle time, and resamples to a fixed duration."""
    if len(data) <= window_size:
        return {"initial_values": initial_values, "data": []}, 0

    data = remove_idle_segments(data, idle_threshold=6)

    # Resample trajectory to match target duration
    data = resample_trajectory(data, target_frames=target_frames)

    # Extract required data fields
    joint_angles = np.array([entry["joint_angles"] for entry in data])
    slab_positions = np.array([entry["slab_position"] for entry in data])
    slab_orientations = np.array([entry["slab_orientation"] for entry in data])
    contact_forces = np.array([entry["contact"]["forces"]["full_contact_force"] for entry in data])

    # Apply smoothing filters
    smoothed_joint_angles = np.apply_along_axis(moving_average, axis=0, arr=joint_angles, window=window_size)
    smoothed_slab_positions = np.apply_along_axis(moving_average, axis=0, arr=slab_positions, window=window_size)
    smoothed_slab_orientations = np.apply_along_axis(moving_average, axis=0, arr=slab_orientations, window=window_size)
    smoothed_contact_forces = np.apply_along_axis(moving_average, axis=0, arr=contact_forces, window=window_size)

    smoothing_effect = compute_difference(joint_angles[:len(smoothed_joint_angles)], smoothed_joint_angles)

    smoothed_data = []
    for i in range(len(smoothed_joint_angles)):
        smoothed_data.append({
            "timestamp": data[i]["timestamp"],
            "joint_angles": smoothed_joint_angles[i].tolist(),
            "slab_position": smoothed_slab_positions[i].tolist(),
            "slab_orientation": smoothed_slab_orientations[i].tolist(),
            "contact": {
                "geom1": data[i]["contact"]["geom1"],
                "geom2": data[i]["contact"]["geom2"],
                "forces": {
                    "normal_force": 0.0,
                    "tangential_force_x": 0.0,
                    "tangential_force_y": 0.0,
                    "full_contact_force": smoothed_contact_forces[i].tolist()
                }
            }
        })

    return {"initial_values": initial_values, "data": smoothed_data}, smoothing_effect

def save_json(data, output_path):
    """Saves processed data to a JSON file."""
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

def process_json(input_file, output_file, window_size=5, target_frames=3000):

    raw_data = load_json(input_file)
    initial_values, structured_data = flatten_json_structure(raw_data)

    original_count = len(structured_data)
    smoothed_data, smoothing_effect = smooth_trajectory_data(initial_values, structured_data, window_size, target_frames)
    
    total_time = smoothed_data["data"][-1]["timestamp"] - smoothed_data["data"][0]["timestamp"]
    save_json(smoothed_data, output_file)

    print(f"Processed JSON saved to: {output_file}")
    print(f"Total trajectory duration after resampling: {total_time:.2f} seconds")
    print(f"Original trajectory entries: {original_count}")
    print(f"Final trajectory entries after resampling: {len(smoothed_data['data'])}")
    print(f"Average smoothing effect on joint angles: {smoothing_effect:.4f}")

if __name__ == "__main__":
    input_file_path = "traj_trace/place_26.json"
    output_file_path = "trajectories/traj_rebalanced39.json"
    process_json(input_file_path, output_file_path, window_size=5, target_frames=5000)
