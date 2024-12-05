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
        self.logs = []

        # Initialize GUI-related attributes
        self.gui_initialized = False
        self.app = None
        self.window = None

        # Initialize slab position
        slab_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.model.jnt_qposadr[slab_joint_id]
        self.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = [1.5, 0.5, 2.01]
        self.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1, 0, 0, 0]

        # Forward simulation to initialize
        mujoco.mj_forward(self.model, self.data)
        self.render()

        slab_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slab_mocap")
        print("Initialized Slab Position:", self.data.xpos[slab_body_id])
        print("Initialized Slab Orientation:", self.data.xquat[slab_body_id])
        self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True

    def render(self):
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def log_contact_forces(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)

            if contact.geom1 == 30 and contact.geom2 == 31:
                print(f"Contact detected between Geom1: {contact.geom1} and Geom2: {contact.geom2}")
                if not self.welded and not self.contact_made:
                    self.welded = True

            if (contact.geom1 == 31 and contact.geom2 == 38) or (contact.geom1 == 31 and contact.geom2 == 36):
                print(f"Contact detected between Geom1: {contact.geom1} and Geom2: {contact.geom2}")
                if self.welded and not self.contact_made:
                    self.welded = False
                    self.contact_made = True
                    print("Contact made, disabling GUI controls.")

    def simulate(self):
        while not self.contact_made:
            self.log_contact_forces()
            mujoco.mj_step(self.model, self.data)
            self.render()

        # Once contact is made, switch to autonomous mode
        self.run_autonomous_simulation()

    def run_autonomous_simulation(self):
        print("Running autonomous simulation...")
        while True:
            mujoco.mj_step(self.model, self.data)
            self.render()

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

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.close_application)
            layout.addWidget(close_button)

            self.window.setLayout(layout)
            self.gui_initialized = True

    def disable_gui(self):
        """Disable GUI elements after contact is made."""
        if self.gui_initialized:
            for slider in self.sliders:
                slider.setEnabled(False)

    def run_gui(self):
        if not self.gui_initialized:
            self.init_gui()
        self.window.show()

        # Monitor simulation state and disable GUI if necessary
        while True:
            self.log_contact_forces()
            if self.contact_made:
                self.disable_gui()
                break
            mujoco.mj_step(self.model, self.data)
            self.render()
            QApplication.processEvents()

        # Switch to autonomous simulation after disabling GUI
        self.run_autonomous_simulation()

    def update_joint_angle(self, joint_index, value):
        if self.contact_made:
            return  # Disable updates once autonomous simulation starts
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
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        env.close()
