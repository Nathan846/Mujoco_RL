from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import sys
import numpy as np
import threading
import mujoco
import mujoco_viewer
import numpy as np
import threading

class MuJoCoSimulation:
    def __init__(self, model_path):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Viewer for real-time rendering
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        # Joint angles
        self.joint_angles = np.zeros(self.model.njnt)

        # Simulation flag
        self.simulating = True

    def update_joint_angle(self, joint_index, angle):
        """Update the joint angle for the specified joint index."""
        self.joint_angles[joint_index] = angle
        self.data.qpos[joint_index] = angle
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        """Run the simulation."""
        while self.simulating:
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()

    def stop(self):
        """Stop the simulation."""
        self.simulating = False
        self.viewer.close()


class SliderApp(QWidget):
    def __init__(self, simulation):
        super().__init__()
        self.simulation = simulation

        # Initialize GUI
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Joint Control Sliders")
        self.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout()
        self.sliders = []

        # Create sliders for each joint
        for i in range(len(self.simulation.joint_angles)):
            h_layout = QHBoxLayout()

            label = QLabel(f"Joint {i + 1}: 0.0 rad")
            h_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-314)  # -3.14 radians
            slider.setMaximum(314)   # 3.14 radians
            slider.setValue(0)
            slider.valueChanged.connect(lambda value, i=i: self.update_joint_angle(i, value, label))
            self.sliders.append(slider)
            h_layout.addWidget(slider)

            layout.addLayout(h_layout)

        # Add a button to stop the simulation
        stop_button = QPushButton("Stop Simulation")
        stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(stop_button)

        self.setLayout(layout)

    def update_joint_angle(self, joint_index, slider_value, label):
        # Convert slider value to radians and update joint angle
        angle = slider_value / 100.0
        label.setText(f"Joint {joint_index + 1}: {angle:.2f} rad")
        self.simulation.update_joint_angle(joint_index, angle)

    def stop_simulation(self):
        self.simulation.stop()
        self.close()


if __name__ == "__main__":
    # Start the MuJoCo simulation in a separate thread
    simulation = MuJoCoSimulation("ur5e.xml")
    simulation_thread = threading.Thread(target=simulation.run)
    simulation_thread.start()

    # Start the slider GUI
    app = QApplication(sys.argv)
    slider_window = SliderApp(simulation)
    slider_window.show()
    sys.exit(app.exec_())

    # Wait for the simulation thread to finish
    simulation_thread.join()
