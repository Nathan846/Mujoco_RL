import os
import json
import torch
from OA_env import OA_env
from tqdm import tqdm

def preprocess_split_by_phase(folder_path, out_q1="expert_p1.pt", out_q2="expert_p2.pt"):
    env = OA_env(render=False)
    q1_transitions = []
    q2_transitions = []

    file_list = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    for fname in tqdm(file_list, desc="Processing expert files"):
        file_path = os.path.join(folder_path, fname)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            actions_with_phase = [
                (e["discrete_action"], e.get("phase", 0))
                for e in data["data"]
                if "discrete_action" in e and e["discrete_action"] is not None
            ]

            obs = env.reset()
            for action, phase in actions_with_phase:
                next_obs, reward, done, _ = env.step(action)
                if reward is None:
                    reward = 0.0

                transition = (obs, action, reward, next_obs, float(done))
                if phase == 0:
                    pass
                    q1_transitions.append(transition)
                elif phase == 1:
                    q2_transitions.append(transition)

                obs = env.reset() if done else next_obs

        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")

    print(f"✅ Q1 transitions (phase=0): {len(q1_transitions)}")
    print(f"✅ Q2 transitions (phase=1): {len(q2_transitions)}")

    torch.save(q1_transitions, out_q1)
    torch.save(q2_transitions, out_q2)
    print(f"💾 Saved to: {out_q1}, {out_q2}")

if __name__ == "__main__":
    preprocess_split_by_phase("processed_trajs_all/")
