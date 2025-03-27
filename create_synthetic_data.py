import json
import time
import copy
import numpy as np
from pathlib import Path

log_path = Path("place_401.json")
with open(log_path, "r") as f:
    log_json = json.load(f)

step_sizes = {
    0: 0.0004363326388885369,
    1: 0.0004363326388885369,
    2: 0.0008726649999999999, 
    3: 0.0004363326388885369,
    4: 0.0004363326388885369,
    5: 0.0004363326388885369,
    6: 0.0004363326388885369
}

def infer_discrete_action(curr, prev, step_sizes):
    for idx, (c, p) in enumerate(zip(curr, prev)):
        diff = round(c - p, 10)
        if abs(diff) < 1e-8:
            continue  

        step = step_sizes[idx]
        num_steps = round(diff / step)

        if num_steps == 0:
            continue

        direction = 0 if num_steps > 0 else 1
        action = 2 * idx + direction
        return action, abs(num_steps)
    return None, 0

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

# === Process all logs ===
all_logs = log_json["data"]
final_logs = [all_logs[0]]
final_logs[0]["discrete_action"] = None  # no action at start

for i in range(1, len(all_logs)):
    prev = all_logs[i - 1]
    curr = all_logs[i]

    action, num_steps = infer_discrete_action(curr["joint_angles"], prev["joint_angles"], step_sizes)

    if action is None:
        curr["discrete_action"] = None
        final_logs.append(curr)
        continue

    # Generate intermediate steps
    synth = generate_synthetic_trajectory(prev, [action] * num_steps, step_sizes)

    # Add action field to synthetic logs
    for s in synth:
        s["discrete_action"] = action

    # Append all synthetic and final log
    final_logs.extend(synth)

# === Save to file ===
out_json = {
    "initial_values": log_json["initial_values"],
    "data": final_logs
}

with open("place_403_expanded.json", "w") as f:
    json.dump(out_json, f, indent=2)

print(f"✅ Generated place_403_expanded.json with {len(final_logs)} entries.")
