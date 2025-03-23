from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np
import mujoco
import mujoco_viewer
import sys
import json
import time
from threading import Thread
from scipy.spatial.transform import Rotation as R
LOG_FILE = "traj_trace/place_40.json"
class MuJoCoEnv:
    def __init__(self, model_path):
        self.integration_dt = 1.0
        self.resolution = 0.05
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
        self.simulation_thread = None
        self.initial_slab_quat = np.array([1, 0, 0, 0])
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        init_quat = [1]
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1,0,0,0]
        self.data.qvel[8:11] = [0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.render()
        # self.print_all_geoms()
        self.logs = {
            "initial_values": {
                "init_pos": slab_pos,
                "init_quat": init_quat
            },
            "data": []
        }

    def render(self):
        if(self.welded and not self.contact_made):
            self.update_slab_to_match_eef()
        mujoco.mj_forward(self.model, self.data)
        try:
            self.viewer.render()
        except:
            self.write_logs(LOG_FILE)
            print(f"saved data to {LOG_FILE}")
            sys.exit(0)
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
        vel = self.data.qvel.tolist()
        slab_pos = self.data.qpos[7:10].tolist()
        slab_quat = self.data.qpos[10:14].tolist()
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        qpos_start_idx = self.model.jnt_qposadr[joint_id]
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # print(contact)
            # print(contact.pos)
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            if contact.geom1 == 30 and contact.geom2 == 31:
                if not self.welded and not self.contact_made:
                    self.welded = True
                continue    
            # if((contact.geom1==31 and contact.geom2 == 36) or (contact.geom1 == 31 and contact.geom2==38)):
            #     # print('contacting')
            #     self.contact_made = True[0.57004024 0.46294754 0.42828982 0.52659427]
                
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
        self.logs["data"].append(log_data)
    def update_slab_to_match_eef(self):
        eef_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        
        eef_pos = self.data.xpos[eef_body_id]
        eef_quat = self.data.xquat[eef_body_id]
        
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = eef_pos
        
        if not hasattr(self, 'eef_contact_quat'):
            self.eef_contact_quat = eef_quat.copy() 
            self.slab_contact_quat = self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7].copy()
        
        delta_quat = self.quaternion_multiply(eef_quat, self.quaternion_inverse(self.eef_contact_quat))
        
        new_slab_quat = self.quaternion_multiply(delta_quat, self.slab_contact_quat)
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = new_slab_quat

        mujoco.mj_forward(self.model, self.data)

    def quaternion_inverse(self, quat):
        """Compute the inverse of a quaternion (conjugate for unit quaternion)."""
        w, x, y, z = quat
        return np.array([w, -x, -y, -z])

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def simulate(self):
        while not self.contact_made:
            self.log_contact_forces()
            mujoco.mj_step(self.model, self.data)
            self.render()

    def check_joint_limits(self, q):
        lower_limits = self.model.jnt_range[:6, 0]
        upper_limits = self.model.jnt_range[:6, 1]
        return np.clip(q, lower_limits, upper_limits)
    def get_all_contact_points(self):
        points_geom_36 = []
        points_geom_38 = []

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == 31 and contact.geom2 == 36) or \
            (contact.geom2 == 31 and contact.geom1 == 36):
                points_geom_36.append(contact.pos)
            elif (contact.geom1 == 31 and contact.geom2 == 38) or \
                (contact.geom2 == 31 and contact.geom1 == 38):
                points_geom_38.append(contact.pos)

        print(f"Points for geom 36: {points_geom_36}")
        print(f"Points for geom 38: {points_geom_38}")

        return points_geom_36, points_geom_38


    def choose_two_points(self, points_geom_36, points_geom_38):
        if not points_geom_36 or not points_geom_38:
            print("One or both lists are empty. Cannot find closest points.")
            return None

        points_geom_36 = np.array(points_geom_36)
        points_geom_38 = np.array(points_geom_38)

        min_dist = float('inf')
        chosen_points = None

        for point_36 in points_geom_36:
            for point_38 in points_geom_38:
                dist = np.linalg.norm(point_36 - point_38)
                if 0.5 <= dist < min_dist:
                    min_dist = dist
                    chosen_points = (point_36, point_38)

        if chosen_points is None:
            print("No points found with the required distance constraint.")
        else:
            print(f"Chosen points: {chosen_points}, Distance: {min_dist}")

        return chosen_points
        
    def init_gui(self):
        if not self.gui_initialized:
            self.app = QApplication(sys.argv)
            self.window = QWidget()
            self.window.setWindowTitle("Joint Angle and Camera Control")
            self.window.setGeometry(100, 100, 400, 600)

            layout = QVBoxLayout()
            self.sliders = []
            self.labels = []
            self.toggle_state = False  

            # Define the resolution for joint angle changes
            self.resolution = 0.05  # or 0.025 for finer control

            for i in range(7):
                h_layout = QHBoxLayout()

                # Create label with fixed width and monospaced font
                label = QLabel(f"Jt {i + 1}: 0.00°")
                label.setFixedWidth(120)  # adjust width as needed for consistency
                label.setStyleSheet("font-family: Courier;")  # use a monospaced font
                self.labels.append(label)
                h_layout.addWidget(label)

                # Set up slider with integer values scaled by resolution
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(int(-180 / self.resolution))
                slider.setMaximum(int(180 / self.resolution))
                slider.setValue(0)
                slider.valueChanged.connect(lambda value, i=i: self.update_joint_angle(i, value * self.resolution))
                self.sliders.append(slider)
                h_layout.addWidget(slider)

                # Decrement button using the resolution
                decrement_button = QPushButton("-")
                decrement_button.clicked.connect(lambda _, i=i: self.adjust_joint_angle(i, -self.resolution))
                h_layout.addWidget(decrement_button)

                # Increment button using the resolution
                increment_button = QPushButton("+")
                increment_button.clicked.connect(lambda _, i=i: self.adjust_joint_angle(i, self.resolution))
                h_layout.addWidget(increment_button)

                layout.addLayout(h_layout)

            self.toggle_button = QPushButton("Toggle OFF")
            self.toggle_button.setCheckable(True)
            self.toggle_button.clicked.connect(self.toggle_button_clicked)
            layout.addWidget(self.toggle_button)

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close_application)
            layout.addWidget(close_button)

            self.window.setLayout(layout)
            self.gui_initialized = True
    def toggle_button_clicked(self):
        self.toggle_state = self.toggle_button.isChecked()
        if self.toggle_state:
            self.toggle_button.setText("Toggle ON")
        else:
            self.toggle_button.setText("Toggle OFF")

    def update_joint_angle(self, joint_index, value):
        angle_rad = np.radians(value)
        self.data.qpos[joint_index] = angle_rad
        mujoco.mj_forward(self.model, self.data)
        self.render()
        self.labels[joint_index].setText(f"Jt {joint_index + 1}: {value:.2f}°")

    def adjust_joint_angle(self, index, increment):
        # Convert the slider's integer value back to degrees using resolution.
        current_value = self.sliders[index].value() * self.resolution
        new_value = current_value + increment
        new_value = max(-180.0, min(180.0, new_value))
        # Set the slider value back as an integer by dividing by resolution.
        self.sliders[index].setValue(int(new_value / self.resolution))

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
            self.render()
            QApplication.processEvents()

    def update_joint_angle(self, joint_index, value):
        if self.contact_made:
            return
        angle_rad = np.radians(value)
        self.data.qpos[joint_index] = angle_rad
        mujoco.mj_forward(self.model, self.data)
        self.render()
        self.labels[joint_index].setText(f"Jt {joint_index + 1}: {value}°")

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