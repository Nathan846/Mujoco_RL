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
def resample_trajectory(data, target_duration=90):
    """Resamples the trajectory to fit the target duration while keeping the shape intact."""
    if len(data) < 2:
        return data  # Not enough data to interpolate

    start_time = data[0]["timestamp"]
    end_time = data[-1]["timestamp"]
    original_duration = end_time - start_time

    if original_duration <= 0:
        return data  # Avoid division by zero if timestamps are wrong

    sampling_rate = len(data) / original_duration  # Entries per second

    # Determine how many entries the resampled trajectory should have
    target_steps = int(sampling_rate * target_duration)  # Keep data density consistent

    # Extract original timestamps
    original_timestamps = np.array([entry["timestamp"] for entry in data])
    
    # Generate new equally spaced timestamps
    new_timestamps = np.linspace(start_time, start_time + target_duration, target_steps)

    # Interpolation function for different fields
    def interpolate_field(field, nested=False):
        """Handles both normal and nested dictionary fields."""
        if nested:
            return interp1d(original_timestamps, np.array([entry[field[0]][field[1]][field[2]] for entry in data]),
                            axis=0, kind='linear', fill_value="extrapolate")(new_timestamps)
        return interp1d(original_timestamps, np.array([entry[field] for entry in data]),
                        axis=0, kind='linear', fill_value="extrapolate")(new_timestamps)

    interpolated_data = []
    for i, t in enumerate(new_timestamps):
        interpolated_data.append({
            "timestamp": float(t),
            "joint_angles": interpolate_field("joint_angles")[i].tolist(),
            "slab_position": interpolate_field("slab_position")[i].tolist(),
            "slab_orientation": interpolate_field("slab_orientation")[i].tolist(),
            "contact": {
                "geom1": data[0]["contact"]["geom1"],  # Use first contact info
                "geom2": data[0]["contact"]["geom2"],
                "forces": {
                    "normal_force": 0.0,
                    "tangential_force_x": 0.0,
                    "tangential_force_y": 0.0,
                    "full_contact_force": interpolate_field(["contact", "forces", "full_contact_force"], nested=True)[i].tolist()
                }
            }
        })

    return interpolated_data

def smooth_trajectory_data(initial_values, data, window_size=5, target_duration=90):
    """Smooths trajectory, removes idle time, and resamples to a fixed duration."""
    if len(data) <= window_size:
        return {"initial_values": initial_values, "data": []}, 0

    data = remove_idle_segments(data, idle_threshold=6)

    # Resample trajectory to match target duration
    data = resample_trajectory(data, target_duration=target_duration)

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

def process_json(input_file, output_file, window_size=5, target_duration=90):

    raw_data = load_json(input_file)
    initial_values, structured_data = flatten_json_structure(raw_data)

    original_count = len(structured_data)
    smoothed_data, smoothing_effect = smooth_trajectory_data(initial_values, structured_data, window_size, target_duration)
    
    total_time = smoothed_data["data"][-1]["timestamp"] - smoothed_data["data"][0]["timestamp"]
    save_json(smoothed_data, output_file)

    print(f"Processed JSON saved to: {output_file}")
    print(f"Total trajectory duration after resampling: {total_time:.2f} seconds")
    print(f"Original trajectory entries: {original_count}")
    print(f"Final trajectory entries after resampling: {len(smoothed_data['data'])}")
    print(f"Average smoothing effect on joint angles: {smoothing_effect:.4f}")

if __name__ == "__main__":
    input_file_path = "trajectories/test1.json"
    output_file_path = "resampled_smoothed_trajectory.json"
    process_json(input_file_path, output_file_path, window_size=5, target_duration=90)
