import json

def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def remove_consecutive_duplicate_frames(frames):
    """
    Given a list of frames (each a dictionary with 'joint_angles'),
    remove consecutive duplicates based on 'joint_angles'.
    """
    if not frames:
        return frames
    
    pruned = [frames[0]]  # Always keep the very first frame
    last_kept_angles = tuple(frames[0]["joint_angles"])

    for frame in frames[1:]:
        current_angles = tuple(frame["joint_angles"])
        if current_angles != last_kept_angles:
            pruned.append(frame)
            last_kept_angles = current_angles
    
    return pruned

def process_json(input_file, output_file):
    data = load_json(input_file)
    # 'data' has the form:
    # {
    #   "initial_values": { ... },
    #   "data": [
    #       [...],   # sublist 1
    #       [...],   # sublist 2
    #       ...
    #   ]
    # }

    initial_values = data.get("initial_values", {})
    all_sub_lists = data.get("data", [])

    # Process each sub-list in "data"
    pruned_data = []
    total_original_frames = 0
    total_pruned_frames = 0

    for sub_list in all_sub_lists:
        original_len = len(sub_list)
        pruned_sub_list = remove_consecutive_duplicate_frames(sub_list)
        pruned_data.append(pruned_sub_list)
        total_original_frames += original_len
        total_pruned_frames += len(pruned_sub_list)

    # Reconstruct final output in the same structure
    output_data = {
        "initial_values": initial_values,
        "data": pruned_data
    }

    with open(output_file, "w") as out:
        json.dump(output_data, out, indent=4)

    print(f"Processed JSON saved to: {output_file}")
    print(f"Total original frames: {total_original_frames}")
    print(f"Total frames after pruning: {total_pruned_frames}")

if __name__ == "__main__":
    input_file_path = "place_100.json"
    output_file_path = "place100prunde.json"
    process_json(input_file_path, output_file_path)
