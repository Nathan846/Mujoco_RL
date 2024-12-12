import mujoco
import mujoco_viewer
import numpy as np
import json
import time

class MuJoCoEnv:
    def __init__(self, model_path, x, y, z):
        self.integration_dt = 1.0
        self.dt = 0.002
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.logs = []

        # Set the slab position
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = [x, y, z]
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = [1, 5, 0, 0]  # Default quaternion

        # Forward the simulation to initialize
        mujoco.mj_forward(self.model, self.data)

        print("Initialized Slab Position:", self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3])
        print("Initialized Slab Orientation:", self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7])

    def render(self):
        self.viewer.render()

    def simulate(self, steps=1000):
        for step in range(steps):
            mujoco.mj_step(self.model, self.data)
            self.log_state()
            self.render()

    def log_state(self):
        slab_pos = self.data.qpos[7:10].tolist()
        slab_quat = self.data.qpos[10:14].tolist()
        log_entry = {
            "timestamp": time.time(),
            "slab_position": slab_pos,
            "slab_orientation": slab_quat
        }
        self.logs.append(log_entry)

    def write_logs(self, filename="simulation_logs.json"):
        with open(filename, "w") as file:
            json.dump(self.logs, file, indent=4)
        print(f"Logs saved to {filename}")

    def close(self):
        self.viewer.close()

if __name__ == "__main__":
    model_path = "universal_robots_ur5e/scene.xml"
    x, y, z = 0, 0, 2.0

    env = MuJoCoEnv(model_path, x, y, z)
    try:
        env.simulate(steps=500)
    finally:
        env.write_logs()
        env.close()
