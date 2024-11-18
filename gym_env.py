from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np
import mujoco
import mujoco_viewer
import sys


class MuJoCoEnv:
    def __init__(self, model_path):
        self.integration_dt = 1.0
        self.damping = 1e-4
        self.gravity_compensation = True
        self.dt = 0.002
        self.max_angvel = 1.0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.model.opt.timestep = self.dt
        self.key_id = self.model.key("home").id
        self.dof_ids = np.arange(self.model.nq)  # Assuming 1 DoF per joint

        self.current_angles = np.zeros(len(self.dof_ids))  # Store current joint angles
        self.gui_initialized = False  # Track if GUI is initialized
        self.app = None
        self.window = None

    def render(self):
        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def init_gui(self):
        if not self.gui_initialized:
            self.app = QApplication(sys.argv)
            self.window = QWidget()
            self.window.setWindowTitle("Joint Angle Control")
            self.window.setGeometry(100, 100, 400, 600)

            layout = QVBoxLayout()
            self.sliders = []
            self.labels = []

            for i in range(len(self.dof_ids)):
                h_layout = QHBoxLayout()

                label = QLabel(f"Joint {i + 1}: 0.0")
                self.labels.append(label)
                h_layout.addWidget(label)

                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(-180)  # Degrees
                slider.setMaximum(180)
                slider.setValue(0)
                slider.valueChanged.connect(lambda value, i=i: self.update_joint_angle(i, value))
                self.sliders.append(slider)
                h_layout.addWidget(slider)

                layout.addLayout(h_layout)

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close_application)
            layout.addWidget(close_button)

            self.window.setLayout(layout)
            self.gui_initialized = True

    def update_joint_angle(self, joint_index, value):
        angle_rad = np.radians(value)
        self.current_angles[joint_index] = angle_rad
        self.data.qpos[:len(self.current_angles)] = self.current_angles
        mujoco.mj_forward(self.model, self.data)
        self.render()

        self.labels[joint_index].setText(f"Joint {joint_index + 1}: {value}°")

    def run_gui(self):
        if not self.gui_initialized:
            self.init_gui()
        self.window.show()
        sys.exit(self.app.exec_())

    def close_application(self):
        self.close()
        self.window.close()


if __name__ == "__main__":
    model_path = "universal_robots_ur5e/scene.xml"
    env = MuJoCoEnv(model_path)

    # Start GUI
    env.run_gui()
