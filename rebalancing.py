import json
import numpy as np

def load_json(file_path):
    """Loads and parses a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

def flatten_json_structure(data):
    """Extracts the initial values and flattens the JSON structure into a single list of data entries."""
    initial_values = data[0] if isinstance(data[0], dict) else None
    structured_data = []

    for entry in data:
        if isinstance(entry, list):  # Extract lists containing actual data
            structured_data.extend(entry)

    return initial_values, structured_data

def moving_average(array, window):
    """Applies a moving average filter to smooth numerical data."""
    return np.convolve(array, np.ones(window) / window, mode='valid')

def smooth_trajectory_data(initial_values, data, window_size=5):
    """Smooths the trajectory data using a moving average filter while retaining initial values."""
    if len(data) <= window_size:
        return {"initial_values": initial_values, "data": []}
    
    smoothed_data = []
    data = data[1:]  # Skip first entry if necessary

    joint_angles = np.array([entry["joint_angles"] for entry in data])
    slab_positions = np.array([entry["slab_position"] for entry in data])
    slab_orientations = np.array([entry["slab_orientation"] for entry in data])
    contact_forces = np.array([entry["contact"]["forces"]["full_contact_force"] for entry in data])

    # Apply smoothing filters
    smoothed_joint_angles = np.apply_along_axis(moving_average, axis=0, arr=joint_angles, window=window_size)
    smoothed_slab_positions = np.apply_along_axis(moving_average, axis=0, arr=slab_positions, window=window_size)
    smoothed_slab_orientations = np.apply_along_axis(moving_average, axis=0, arr=slab_orientations, window=window_size)
    smoothed_contact_forces = np.apply_along_axis(moving_average, axis=0, arr=contact_forces, window=window_size)

    # Reconstruct smoothed trajectory data
    for i in range(len(smoothed_joint_angles)):
        smoothed_entry = {
            "timestamp": data[i + window_size - 1]["timestamp"],
            "joint_angles": smoothed_joint_angles[i].tolist(),
            "slab_position": smoothed_slab_positions[i].tolist(),
            "slab_orientation": smoothed_slab_orientations[i].tolist(),
            "contact": {
                "geom1": data[i + window_size - 1]["contact"]["geom1"],
                "geom2": data[i + window_size - 1]["contact"]["geom2"],
                "forces": {
                    "normal_force": 0.0,
                    "tangential_force_x": 0.0,
                    "tangential_force_y": 0.0,
                    "full_contact_force": smoothed_contact_forces[i].tolist()
                }
            }
        }
        smoothed_data.append(smoothed_entry)

    return {"initial_values": initial_values, "data": smoothed_data}

def filter_zero_joint_angles(data):
    """Removes entries where all joint angles are zero."""
    data["data"] = [entry for entry in data["data"] if any(angle != 0 for angle in entry["joint_angles"])]
    return data

def save_json(data, output_path):
    """Saves processed data to a JSON file."""
    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)

def process_json(input_file, output_file, window_size=5):
    """Main function to process a JSON file, apply smoothing, and save the result."""
    raw_data = load_json(input_file)
    initial_values, structured_data = flatten_json_structure(raw_data)
    smoothed_data = smooth_trajectory_data(initial_values, structured_data, window_size)
    filtered_data = filter_zero_joint_angles(smoothed_data)
    save_json(filtered_data, output_file)

# Example usage:
if __name__ == "__main__":
    input_file_path = "trajectories/gentle_place/place3.json"  # Change this to your input file path
    output_file_path = "filtered_smoothed_place20.json"
    process_json(input_file_path, output_file_path, window_size=5)
    print(f"Processed JSON saved to: {output_file_path}")
