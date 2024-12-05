from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np
import mujoco
import mujoco_viewer
import sys
import json
import time
class MuJoCoEnv:
    def __init__(self, model_path):
        self.integration_dt = 1.0
        self.damping = 1e-4
        self.gravity_compensation = True
        self.dt = 0.002
        self.contact_made = False
        self.welded = False
        self.max_angvel = 1.0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.key_id = self.model.key("home").id
        self.dof_ids = np.arange(self.model.nq)
        self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.current_angles = np.zeros(len(self.dof_ids))
        self.last_contact_force = []
        self.gui_initialized = False
        self.app = None
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            # print(f"Body ID: {body_id}, Body Name: {body_name}")
        self.window = None
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        self.logs = []
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = [1.5, 0.5, 2.01] 
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = [1, 0, 0, 0] 
        mujoco.mj_forward(self.model, self.data)
        self.render()
        # self.print_all_joints()
        # self.print_all_geoms()
        slab_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
        print("Initialized Slab Position:", self.data.xpos[slab_body_id])
        print("Initialized Slab Orientation:", self.data.xquat[slab_body_id])
        self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
    def render(self):
        mujoco.mj_forward(self.model, self.data)
        self.append_logs()
        self.log_contact_forces()
        if(self.welded):
            self.update_slab_to_match_eef()
        self.viewer.render()
    def print_all_joints(self):
        print("\n--- Joint Information ---")
        for joint_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            
            qpos_start_idx = self.model.jnt_qposadr[joint_id]
            qpos_size = self.model.jnt_dofadr[joint_id] - qpos_start_idx

            print(f"Joint ID: {joint_id}, Name: {joint_name}")
            print(f"  Position (qpos): {self.data.qpos[qpos_start_idx:qpos_start_idx + qpos_size]}")
            print(f"  Velocity (qvel): {self.data.qvel[qpos_start_idx:qpos_start_idx + qpos_size]}")
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

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def move_camera(self, azimuth=None, elevation=None, distance=None):
        """Move the camera programmatically."""
        if azimuth is not None:
            self.viewer.cam.azimuth = azimuth
        if elevation is not None:
            self.viewer.cam.elevation = elevation
        if distance is not None:
            self.viewer.cam.distance = distance

    def init_gui(self):
        if not self.gui_initialized:
            print("cute")
            self.app = QApplication(sys.argv)
            self.window = QWidget()
            self.window.setWindowTitle("Joint Angle and Camera Control")
            self.window.setGeometry(100, 100, 400, 600)

            layout = QVBoxLayout()
            self.sliders = []
            self.labels = []

            for i in range(7):
                h_layout = QHBoxLayout()

                label = QLabel(f"Joint {i + 1}: 0.0")
                self.labels.append(label)
                h_layout.addWidget(label)

                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(-180)
                slider.setMaximum(180)
                slider.setValue(0)
                slider.valueChanged.connect(lambda value, i=i: self.update_joint_angle(i, value))
                self.sliders.append(slider)
                h_layout.addWidget(slider)

                layout.addLayout(h_layout)

            cam_layout = QVBoxLayout()
            self.cam_labels = {
                "azimuth": QLabel("Camera Azimuth: 0"),
                "elevation": QLabel("Camera Elevation: 0"),
                "distance": QLabel("Camera Distance: 0"),
            }
            self.cam_sliders = {
                "azimuth": QSlider(Qt.Horizontal),
                "elevation": QSlider(Qt.Horizontal),
                "distance": QSlider(Qt.Horizontal),
            }

            self.cam_sliders["azimuth"].setMinimum(-180)
            self.cam_sliders["azimuth"].setMaximum(180)
            self.cam_sliders["azimuth"].valueChanged.connect(self.update_camera_azimuth)

            self.cam_sliders["elevation"].setMinimum(-90)
            self.cam_sliders["elevation"].setMaximum(90)
            self.cam_sliders["elevation"].valueChanged.connect(self.update_camera_elevation)

            self.cam_sliders["distance"].setMinimum(1)
            self.cam_sliders["distance"].setMaximum(10)
            self.cam_sliders["distance"].valueChanged.connect(self.update_camera_distance)

            for key in self.cam_sliders:
                cam_layout.addWidget(self.cam_labels[key])
                cam_layout.addWidget(self.cam_sliders[key])

            layout.addLayout(cam_layout)

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close_application)
            layout.addWidget(close_button)

            self.window.setLayout(layout)
            self.gui_initialized = True
    def append_logs(self):
        joint_angles = self.data.qpos[:7].tolist()
        
        slab_pos = self.data.qpos[7:10].tolist()
        slab_quat = self.data.qpos[10:14].tolist()

        # Create log entry
        log_entry = {
            "timestamp": time.time(),
            "joint_angles": joint_angles,
            "slab_position": slab_pos,
            "slab_orientation": slab_quat,
            "contact": self.last_contact_force
        }

        self.logs.append(log_entry)
    def update_joint_angle(self, joint_index, value):
        if(joint_index>=7 or self.contact_made):
            return
        angle_rad = np.radians(value)
        
        self.current_angles[joint_index] = angle_rad
        current_pos = self.data.qpos
        slab_pos = self.data.qpos[7:10].copy()  
        slab_quat = self.data.qpos[10:14].copy()  
        self.data.qpos[:len(self.current_angles)] = self.current_angles
        
        mujoco.mj_forward(self.model, self.data)
        
        self.render()
        

        self.labels[joint_index].setText(f"Joint {joint_index + 1}: {value}°")
    def update_slab_to_match_eef(self):
        eef_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]

        eef_pos = self.data.xpos[eef_body_id]
        eef_quat = self.data.xquat[eef_body_id]

        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx+3] = eef_pos
        self.data.qpos[slab_qpos_start_idx+3:slab_qpos_start_idx+7] = eef_quat

        mujoco.mj_forward(self.model, self.data)


    def log_contact_forces(self):
        log_data = []  

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            joint_angles = self.data.qpos[:7].tolist()
            slab_pos = self.data.qpos[7:10].tolist()
            slab_quat = self.data.qpos[10:14].tolist()
            
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
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

            if contact.geom1 == 30 and contact.geom2 == 31:
                print(f"Contact detected between Geom1: {contact.geom1} and Geom2: {contact.geom2}")
                
                if not self.welded and not self.contact_made:
                    self.welded = True
            if (contact.geom1 == 31 and contact.geom2 == 38) or (contact.geom1 == 31 and contact.geom2 == 36):
                print(f"Contact detected between Geom1: {contact.geom1} and Geom2: {contact.geom2}")
                
                if self.welded and not self.contact_made:
                    self.welded = False
                    self.contact_made = True
                slab_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
                
                small_velocity = np.array([0.01, 0.01, 0])
                self.data.qvel[8:11] = small_velocity

    def update_camera_azimuth(self, value):
        self.move_camera(azimuth=value)
        self.cam_labels["azimuth"].setText(f"Camera Azimuth: {value}")
        self.render()

    def update_camera_elevation(self, value):
        self.move_camera(elevation=value)
        self.cam_labels["elevation"].setText(f"Camera Elevation: {value}")
        self.render()

    def update_camera_distance(self, value):
        self.move_camera(distance=value)
        self.cam_labels["distance"].setText(f"Camera Distance: {value}")
        self.render()

    def run_gui(self):
        if not self.gui_initialized:
            self.init_gui()
        self.window.show()
        sys.exit(self.app.exec_())

    def close_application(self):
        self.write_logs(filename="contact_angle_success2.json")
        self.close()
        self.window.close()
    def write_logs(self, filename="cf_log.json"):
        with open(filename, "w") as file:
            json.dump(self.logs, file, indent=4)
        print(f"Logs saved to {filename}")

if __name__ == "__main__":
    model_path = "universal_robots_ur5e/scene.xml"
    env = MuJoCoEnv(model_path)

    env.run_gui()
