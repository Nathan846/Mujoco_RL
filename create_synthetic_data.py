import json
import time
import copy
import numpy as np
from pathlib import Path

# === Step Sizes Per Joint ===
step_sizes = {
    0: 0.0004363326388885369,
    1: 0.0004363326388885369,
    2: 0.0008726649999999999,
    3: 0.0004363326388885369,
    4: 0.0004363326388885369,
    5: 0.0004363326388885369,
    6: 0.0004363326388885369
}

def infer_discrete_actions(curr, prev, step_sizes, residuals):
    actions = []
    for idx, (c, p) in enumerate(zip(curr, prev)):
        diff = c - p + residuals.get(idx, 0.0)
        step = step_sizes[idx]
        num_steps = int(diff / step)
        residuals[idx] = diff - (num_steps * step)
        if num_steps == 0:
            continue
        direction = 0 if num_steps > 0 else 1
        action = 2 * idx + direction
        actions.append((action, abs(num_steps)))
    return actions

def generate_synthetic_trajectory(base_entry, action_sequence, joint_step_sizes):
    synthetic_entries = []
    prev_entry = copy.deepcopy(base_entry)
    prev_angles = prev_entry["joint_angles"]
    timestamp = prev_entry["timestamp"]
    for action in action_sequence:
        joint_idx = action // 2
        sign = 1 if action % 2 == 0 else -1
        delta = joint_step_sizes.get(joint_idx, 0.002) * sign
        new_entry = copy.deepcopy(prev_entry)
        new_joint_angles = copy.deepcopy(prev_angles)
        new_joint_angles[joint_idx] += delta
        new_entry["prev_angles"] = copy.deepcopy(prev_angles)
        new_entry["joint_angles"] = new_joint_angles
        timestamp += 0.01
        new_entry["timestamp"] = timestamp
        new_entry["discrete_action"] = action
        synthetic_entries.append(new_entry)
        prev_angles = new_joint_angles
        prev_entry = new_entry
    return synthetic_entries

def slab_changed(pos1, pos2, quat1, quat2, pos_thresh=1e-4, quat_thresh=1e-3):
    pos1, pos2 = np.array(pos1), np.array(pos2)
    quat1, quat2 = np.array(quat1), np.array(quat2)
    pos_dist = np.linalg.norm(pos1 - pos2)
    quat_dot = np.clip(np.dot(quat1 / np.linalg.norm(quat1), quat2 / np.linalg.norm(quat2)), -1.0, 1.0)
    quat_diff = 2 * np.arccos(abs(quat_dot))
    return pos_dist > pos_thresh or quat_diff > quat_thresh

# === Loop over folder of files ===
input_folder = Path("traj_files")  # 👈 Update this
output_suffix = "_updated.json"
json_files = list(input_folder.glob("*.json"))

print(f"🔍 Found {len(json_files)} JSON files in {input_folder}")

for file_path in json_files:
    print(file_path)
    with open(file_path, "r") as f:
        log_json = json.load(f)
    if isinstance(log_json, list):
        init_vals = log_json[0]
        all_logs = []
        for chunk in log_json[1:]:
            if isinstance(chunk, list):
                all_logs.extend(chunk)
            else:
                print(f"⚠️ Unexpected non-list segment in {file_path.name}, skipping: {chunk}")
    
    elif isinstance(log_json, dict):
        init_vals = log_json.get("initial_values", {})
        data = log_json.get("data", [])
        if isinstance(data, list) and isinstance(data[0], list):
            all_logs = []
            for chunk in data:
                all_logs.extend(chunk)
        else:
            all_logs = data

    else:
        print(log_json)
        raise ValueError(f"Unexpected format in file: {file_path.name}")

    if not all_logs:
        print(f"⚠️ Empty data in {file_path.name}, skipping.")
        continue

    residuals = {}
    final_logs = [all_logs[0]]
    final_logs[0]["discrete_action"] = None
    final_logs[0]["phase"] = 0

    prev = all_logs[0]
    phase = 0

    for i in range(1, len(all_logs)):
        curr = all_logs[i]

        if phase == 0 and slab_changed(
            prev["slab_position"], curr["slab_position"],
            prev["slab_orientation"], curr["slab_orientation"]
        ):
            print(f"🔁 Phase transition at step {i} in {file_path.name}")
            phase = 1

        curr["phase"] = phase

        actions = infer_discrete_actions(curr["joint_angles"], prev["joint_angles"], step_sizes, residuals)

        if not actions:
            curr["discrete_action"] = None
            final_logs.append(curr)
            prev = curr
            continue

        for action, num_steps in actions:
            synth = generate_synthetic_trajectory(prev, [action] * num_steps, step_sizes)
            for s in synth:
                s["discrete_action"] = action
                s["phase"] = phase
            final_logs.extend(synth)
            prev = synth[-1] if synth else prev

        curr["discrete_action"] = None
        final_logs.append(curr)
        prev = curr

    out_json = {
        "initial_values": init_vals,
        "data": final_logs
    }
    output_folder = Path("processed_trajs")  # 👈 change folder name here
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / (file_path.stem + output_suffix)
    with open(output_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"✅ Processed: {file_path.name} → {output_path.name} ({len(final_logs)} entries)")
