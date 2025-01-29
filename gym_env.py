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
LOG_FILE = "trajectories/test1.json"
class MuJoCoEnv:
    def __init__(self, model_path):
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
        self.simulation_thread = None
        self.eef_slab_quat_offset = None
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]
        self.eef_slab_pos_offset = None
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
            if(self.toggle_state):
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
        """Ensure slab follows EEF translation while maintaining proportional rotation."""
        eef_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]

        eef_pos = self.data.xpos[eef_body_id]
        eef_quat = self.data.xquat[eef_body_id]
        slab_pos = self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3]
        slab_quat = self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7]

        # **On first contact, store the offset**
        if self.eef_slab_quat_offset is None or self.eef_slab_pos_offset is None:
            self.eef_slab_quat_offset = self.compute_quat_difference(eef_quat, slab_quat)
            self.eef_slab_pos_offset = slab_pos - eef_pos  # Store relative position

        # **Update slab translation (xpos) to follow EEF's motion**
        new_slab_pos = eef_pos + self.eef_slab_pos_offset  # Maintain relative offset

        # **Update slab orientation proportionally**
        new_slab_quat = self.quaternion_multiply(eef_quat, self.eef_slab_quat_offset)

        # Apply new position & rotation to slab
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = new_slab_pos
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = new_slab_quat

        mujoco.mj_forward(self.model, self.data)
    def compute_quat_difference(self, q1, q2):
        """Computes the quaternion difference between two orientations."""
        q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])  # Inverse of q1
        return self.quaternion_multiply(q1_inv, q2)  # q_diff = q1⁻¹ * q2

    def simulate(self):
        while not self.contact_made:
            self.log_contact_forces()
            mujoco.mj_step(self.model, self.data)
            self.render()
        self.start_decoupled_rendering()

    def start_decoupled_rendering(self):
        self.welded = False
        self.contact_made = True
        self.dt = 0.0002
        self.logswritten = False
        axis = self.compute_rotation_axis_from_multiple_contacts()
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        slab_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
        tol = 1e-3
        step_size = 0.01
        damping = 1e-4
        nv = self.model.nv
        while True:
            if not self.final_contact:
                self.rotate_free_joint(axis)
                mujoco.mj_step(self.model, self.data)

                # Get current position of the slab
                slab_pos = self.data.xpos[slab_id]

                # Set the target position for the EEF
                goal = slab_pos

                current_pos = self.data.xpos[eef_id]
                error = np.subtract(goal, current_pos)

                if np.linalg.norm(error) >= tol:
                    jacp = np.zeros((3, nv))
                    jacr = np.zeros((3, nv))  # Rotational Jacobian (not used here)

                    # Calculate the Jacobian for the EEF
                    mujoco.mj_jac(self.model, self.data, jacp, jacr, goal, eef_id)

                    # Calculate delta joint angles for the first 6 joints
                    jacp_6 = jacp[:, :6]

                    n = jacp_6.shape[1]
                    I = np.identity(n)
                    product = jacp_6.T @ jacp_6 + damping * I
                    if np.isclose(np.linalg.det(product), 0):
                        j_inv = np.linalg.pinv(product) @ jacp_6.T
                    else:
                        j_inv = np.linalg.inv(product) @ jacp_6.T

                    delta_q = j_inv @ error

                    q = self.data.qpos[:6].copy()
                    q += step_size * delta_q

                    q = self.check_joint_limits(q)

                    self.data.qpos[:6] = q
                    self.data.ctrl[:6] = q  # Assuming direct control of joint positions
                    mujoco.mj_forward(self.model, self.data)

                # Write logs
                if not self.logswritten:
                    self.write_logs(LOG_FILE)
                    self.logswritten = True            
                self.viewer.render()
    def apply_mujoco_ik(self, target_pos, eef_id):
        # Copy the current state
        # print(target_pos,self.data.mocap_pos[eef_id][:3])
        
        # Set the target position in the mocap body associated with the end effector
        self.data.mocap_pos[eef_id][:3] = target_pos
        
        # Use MuJoCo's inverse kinematics to compute joint positions
        mujoco.mj_inverse(self.model, self.data)
    def check_joint_limits(self, q):
        lower_limits = self.model.jnt_range[:6, 0]  # Lower limits for first 6 joints
        upper_limits = self.model.jnt_range[:6, 1]  # Upper limits for first 6 joints
        return np.clip(q, lower_limits, upper_limits)
    def get_all_contact_points(self):
        """
        Extracts contact points from the MuJoCo contact list and separates them into two lists:
        - One for contacts involving geom1 == 31 and geom2 == 36
        - Another for contacts involving geom1 == 31 and geom2 == 38

        :return: Two lists of contact positions, one for geom2 == 36 and another for geom2 == 38
        """
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
        """
        Finds the closest two points, one from each of the two lists, where the distance between
        them is at least 0.5.

        :param points_geom_36: List of points associated with geom2 == 36
        :param points_geom_38: List of points associated with geom2 == 38
        :return: Tuple of two points (point1, point2) or None if no suitable points
        """
        if not points_geom_36 or not points_geom_38:
            print("One or both lists are empty. Cannot find closest points.")
            return None

        points_geom_36 = np.array(points_geom_36)
        points_geom_38 = np.array(points_geom_38)

        min_dist = float('inf')
        chosen_points = None

        # Iterate over all pairs of points from the two lists
        for point_36 in points_geom_36:
            for point_38 in points_geom_38:
                dist = np.linalg.norm(point_36 - point_38)
                if 0.5 <= dist < min_dist:  # Ensure distance is at least 0.5
                    min_dist = dist
                    chosen_points = (point_36, point_38)

        if chosen_points is None:
            print("No points found with the required distance constraint.")
        else:
            print(f"Chosen points: {chosen_points}, Distance: {min_dist}")

        return chosen_points
    def compute_rotation_axis_from_multiple_contacts(self):
        contact_points_1,contact_points_2 = self.get_all_contact_points()

        point1, point2 = self.choose_two_points(contact_points_1,contact_points_2)
        if point1 is None or point2 is None:
            return None
        axis = point2 - point1
        norm = np.linalg.norm(axis)
        if norm == 0:
            print("Selected contact points are identical; cannot define a rotation axis.")
            return None

        return axis / norm
    
    def rotate_free_joint(self, axis):
        if(axis is None):
            return
        theta = np.radians(0.02)
        w_r = np.cos(theta / 2)

        x_r, y_r, z_r = axis * np.sin(theta / 2)

        q_r = np.array([w_r, x_r, y_r, z_r])
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_mocap")
        qpos_start_idx = self.model.jnt_qposadr[joint_id]
        my_quat = self.data.qpos[qpos_start_idx + 3:qpos_start_idx + 7]
        new_quat = self.quaternion_multiply(my_quat, q_r)
        self.data.qpos[qpos_start_idx + 3:qpos_start_idx + 7] = new_quat
    
    def quaternion_multiply(self,q1, q2):

        w1, x1, y1, z1 = q1

        w2, x2, y2, z2 = q2

        return np.array([

            w1*w2 - x1*x2 - y1*y2 - z1*z2,

            w1*x2 + x1*w2 + y1*z2 - z1*y2,

            w1*y2 - x1*z2 + y1*w2 + z1*x2,

            w1*z2 + x1*y2 - y1*x2 + z1*w2

        ])

    def init_gui(self):
        if not self.gui_initialized:
            self.app = QApplication(sys.argv)
            self.window = QWidget()
            self.window.setWindowTitle("Joint Angle and Camera Control")
            self.window.setGeometry(100, 100, 400, 600)

            layout = QVBoxLayout()
            self.sliders = []
            self.labels = []
            self.toggle_state = False  # Variable to track toggle state

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

            # Toggle button
            self.toggle_button = QPushButton("Toggle OFF")
            self.toggle_button.setCheckable(True)  # Make it toggleable
            self.toggle_button.clicked.connect(self.toggle_button_clicked)
            layout.addWidget(self.toggle_button)

            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close_application)
            layout.addWidget(close_button)

            self.window.setLayout(layout)
            self.gui_initialized = True

    def toggle_button_clicked(self):
        """
        Handler for the toggle button. Updates the toggle state and button text.
        """
        self.toggle_state = self.toggle_button.isChecked()  # Update state
        if self.toggle_state:
            self.toggle_button.setText("Toggle ON")
        else:
            self.toggle_button.setText("Toggle OFF")

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