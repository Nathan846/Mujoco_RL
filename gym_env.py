from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np
import mujoco
import mujoco_viewer
import sys
import json
import time
from threading import Thread

class MuJoCoEnv:
    def __init__(self, model_path):
        # Initialization as before...
        self.integration_dt = 1.0
        self.damping = 1e-4
        self.gravity_compensation = True
        self.dt = 0.00002
        self.contact_made = False
        self.welded = False
        self.max_angvel = 1.0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.logs = []
        self.final_contact = False
        self.gui_initialized = False
        self.app = None
        self.window = None
        self.simulation_thread = None  # Thread for decoupled simulation

        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        init_quat = [1]
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1,0,0,0]
        self.data.qvel[8:11] = [0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.render()
        self.print_all_geoms()
        self.logs = [{"init_pos":slab_pos,"init_quat":init_quat}]


    def render(self):
        if(self.welded and not self.contact_made):
            self.update_slab_to_match_eef()
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()
    def print_all_geoms(self):
        print("\n--- Geom Information ---")
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            
            geom_type = self.model.geom_type[geom_id]
            geom_size = self.model.geom_size[geom_id]
            geom_pos = self.model.geom_pos[geom_id]
            geom_quat = self.model.geom_quat[geom_id]

            print(f"Geom ID: {geom_id}, Name: {geom_name}")
            print(f"  Type: {geom_type}")
            print(f"  Size: {geom_size}")
            print(f"  Position: {geom_pos}")
            print(f"  Orientation (Quat): {geom_quat}")

    def log_contact_forces(self):
        duplicate_count = sum(
            1 for i in range(self.data.ncon) 
            if (self.data.contact[i].geom1 == 31 and self.data.contact[i].geom2 == 36) or
            (self.data.contact[i].geom1 == 36 and self.data.contact[i].geom2 == 31)
        )
        log_data = []  
        joint_angles = self.data.qpos[:7].tolist()
        slab_pos = self.data.qpos[7:10].tolist()
        slab_quat = self.data.qpos[10:14].tolist()

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            if contact.geom1 == 30 and contact.geom2 == 31:
                if not self.welded and not self.contact_made:
                    self.welded = True
            # if((contact.geom1==31 and contact.geom2 == 36) or (contact.geom1 == 31 and contact.geom2==38)):
            #     # print('contacting')
            #     self.contact_made = True
            axis = self.compute_rotation_axis_from_multiple_contacts()
            if axis is not None:
                print("new phase")
                self.contact_made = True
            # if (contact.geom1 == 31 and contact.geom2 == 37):
            #     print(f"Contact detected between Geom1: {contact.geom1} and Geom2: {contact.geom2}")
            #     if self.welded and not self.contact_made:
            #         self.welded = False
            #         self.contact_made = True
            #         print("Contact made, switching to decoupled rendering.")
            contact_force_list = contact_force.tolist()

            log_entry = {
                "timestamp": time.time(),
                "joint_angles": joint_angles,
                "slab_position": slab_pos,
                "slab_orientation": slab_quat,
                "contact": {
                    "geom1": contact.geom1,
                    "geom2": contact.geom2,
                    "forces": {
                        "normal_force": contact_force[0],
                        "tangential_force_x": contact_force[1],
                        "tangential_force_y": contact_force[2],
                        "full_contact_force": contact_force_list
                    }
                }
            }
            log_data.append(log_entry)
        self.logs.append(log_data)
    def update_slab_to_match_eef(self):
        eef_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        eef_pos = self.data.xpos[eef_body_id]
        eef_quat = self.data.xquat[eef_body_id]
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = eef_pos
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = eef_quat

        mujoco.mj_forward(self.model, self.data)

    def simulate(self):
        while not self.contact_made:
            self.log_contact_forces()
            mujoco.mj_step(self.model, self.data)
            self.render()
        self.start_decoupled_rendering()

    def start_decoupled_rendering(self):
        self.welded = False
        self.contact_made = True
        self.dt = 0.02
        self.logswritten = False
        while True:
            if(not self.final_contact):
                mujoco.mj_step(self.model, self.data)
                self.log_contact_forces()
                axis = self.compute_rotation_axis_from_multiple_contacts()
                print(axis)
                self.rotate_free_joint(axis)
                if(not self.logswritten):
                    self.write_logs("new_angle_config.json")
                    self.logswritten = True
            self.viewer.render()

    def get_all_contact_points(self):
        contact_points = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == 31 and contact.geom2 in [36, 38]) or (contact.geom2 == 31 and contact.geom1 in [36, 38]):
                contact_points.append(contact.pos)
        return contact_points
    def choose_two_points(self,contact_points):
        if len(contact_points) < 2:
            print("Not enough contact points to define an axis.")
            return None
        contact_points = np.array(contact_points)
        max_dist = 0
        chosen_points = None
        for i in range(len(contact_points)):
            for j in range(i + 1, len(contact_points)):
                dist = np.linalg.norm(contact_points[i] - contact_points[j])
                if dist > max_dist:
                    max_dist = dist
                    chosen_points = (contact_points[i], contact_points[j])

        return chosen_points
    def compute_rotation_axis_from_multiple_contacts(self):
        contact_points = self.get_all_contact_points()
        if len(contact_points) < 2:
            print("Not enough contact points to compute a rotation axis.")
            return None

        point1, point2 = self.choose_two_points(contact_points)
        if point1 is None or point2 is None:
            return None

        # Compute the axis
        axis = point2 - point1
        norm = np.linalg.norm(axis)
        if norm == 0:
            print("Selected contact points are identical; cannot define a rotation axis.")
            return None

        return axis / norm
    def rotate_free_joint(self, axis):
        """
        Rotate a free joint using the provided quaternion.
        :param model: MuJoCo model object.
        :param data: MuJoCo data object.
        :param joint_name: Name of the free joint.
        :param quaternion: Desired rotation quaternion [w, x, y, z].
        """
        if(axis is None):
            return
        angle = np.radians(30)
        w = np.cos(angle / 2)
        x, y, z = np.sin(angle / 2) * axis
        quat =  np.array([w, x, y, z])
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        qpos_start_idx = self.model.jnt_qposadr[joint_id]
        
        self.data.qpos[qpos_start_idx + 3:qpos_start_idx + 7] = quat

    def init_gui(self):
        if not self.gui_initialized:
            self.app = QApplication(sys.argv)
            self.window = QWidget()
            self.window.setWindowTitle("Joint Angle and Camera Control")
            self.window.setGeometry(100, 100, 400, 600)

            layout = QVBoxLayout()
            self.sliders = []
            self.labels = []

            for i in range(7):
                h_layout = QHBoxLayout()

                # Label for the joint angle
                label = QLabel(f"Joint {i + 1}: 0.0")
                self.labels.append(label)
                h_layout.addWidget(label)

                # Slider for the joint angle
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(-1800)  # -180 scaled by 10
                slider.setMaximum(1800)  # 180 scaled by 10
                slider.setValue(0)
                slider.valueChanged.connect(lambda value, i=i: self.update_joint_angle(i, value / 10.0))
                self.sliders.append(slider)

                h_layout.addWidget(slider)

                # Buttons for fine control
                decrement_button = QPushButton("-")
                decrement_button.clicked.connect(lambda _, i=i: self.adjust_joint_angle(i, -0.1))
                h_layout.addWidget(decrement_button)

                increment_button = QPushButton("+")
                increment_button.clicked.connect(lambda _, i=i: self.adjust_joint_angle(i, 0.1))
                h_layout.addWidget(increment_button)

                layout.addLayout(h_layout)

            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close_application)
            layout.addWidget(close_button)

            self.window.setLayout(layout)
            self.gui_initialized = True

    def update_joint_angle(self, index, value):
        self.labels[index].setText(f"Joint {index + 1}: {value:.1f}")

    def adjust_joint_angle(self, index, increment):
        current_value = self.sliders[index].value() / 10.0  # Convert to float
        new_value = current_value + increment
        new_value = max(-180.0, min(180.0, new_value))  # Clamp value between -180 and 180
        self.sliders[index].setValue(int(new_value * 10))  # Convert back to slider scale
    def write_logs(self, filename="cf_log.json"):
            with open(filename, "w") as file:
                json.dump(self.logs, file, indent=4)
            print(f"Logs saved to {filename}")
    def disable_gui(self):
        """Disable GUI elements after contact is made."""
        if self.gui_initialized:
            for slider in self.sliders:
                slider.setEnabled(False)

    def run_gui(self):
        if not self.gui_initialized:
            self.init_gui()
        self.window.show()

        while True:
            self.log_contact_forces()
            if self.contact_made:
                self.disable_gui()
                break
            self.render()
            QApplication.processEvents()

    def update_joint_angle(self, joint_index, value):
        if self.contact_made:
            return
        angle_rad = np.radians(value)
        self.data.qpos[joint_index] = angle_rad
        mujoco.mj_forward(self.model, self.data)
        self.render()
        self.labels[joint_index].setText(f"Joint {joint_index + 1}: {value}°")

    def close_application(self):
        print("Closing application...")
        self.close()
        self.window.close()

    def close(self):
        self.viewer.close()

if __name__ == "__main__":
    model_path = "universal_robots_ur5e/scene.xml"
    env = MuJoCoEnv(model_path)

    try:
        env.run_gui()
        env.simulate()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        env.close()
    print("Successfully saved log")