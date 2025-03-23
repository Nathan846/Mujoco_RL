import mujoco
import mujoco_viewer
import time
import json

class ReExecuteTrajectory:
    def __init__(self, model_path, trajectory_file):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.trajectory, self.init_state = self.load_trajectory(trajectory_file)

    def load_trajectory(self, file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        if "initial_values" not in data or "data" not in data:
            raise ValueError("Invalid trajectory format. Expected 'initial_values' and 'data' keys.")

        init_state = data["initial_values"]
        trajectory_data = data["data"]

        if not isinstance(trajectory_data, list) or len(trajectory_data) == 0:
            raise ValueError("Invalid trajectory format. Expected 'data' to be a non-empty list.")

        return trajectory_data, init_state

    def set_initial_state(self):
        slab_position = self.init_state["init_pos"]
        slab_orientation = self.init_state["init_quat"] + [0, 0, 0]
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_position
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = slab_orientation

        mujoco.mj_forward(self.model, self.data)

    def execute_trajectory(self):
        """Execute the trajectory by iterating through states."""
        big_traj = [item for sublist in self.trajectory for item in sublist]

        for state in big_traj:
            print(state)
            joint_angles = state["joint_angles"]
            slab_position = state["slab_position"]
            slab_orientation = state["slab_orientation"]

            for i in range(len(joint_angles)):
                self.data.qpos[i] = joint_angles[i]

            self.set_slab_state(slab_position, slab_orientation)

            mujoco.mj_forward(self.model, self.data)
            self.viewer.render()

            time.sleep(0.02)

    def set_slab_state(self, slab_position, slab_orientation):
        """Set the slab state in the simulation."""
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_position
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = slab_orientation

    def close(self):
        """Close the MuJoCo viewer."""
        self.viewer.close()

if __name__ == "__main__":
    model_path = "universal_robots_ur5e/scene.xml"
    trajectory_file = "traj_trace/place_40.json"

    executor = ReExecuteTrajectory(model_path, trajectory_file)
    try:
        executor.set_initial_state()
        executor.execute_trajectory()
    except KeyboardInterrupt:
        print("Execution interrupted.")
    finally:
        executor.close()
