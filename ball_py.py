import mujoco
import numpy as np
import glfw

# Load the model (replace with the path to your XML model file)
model = mujoco.MjModel.from_xml_path("ball_model.xml")

# Create the simulation object (formerly `MjSim`, now `Sim`)
sim = mujoco.sim(model)

# Set initial velocities (linear and angular)
# Linear velocity: [vx, vy, vz]
# Angular velocity: [wx, wy, wz]
sim.data.qvel[:3] = np.array([1.0, 0.0, 0.0])  # Linear velocity in x-direction
sim.data.qvel[3:6] = np.array([0.0, 0.0, 0.5])  # Angular velocity around z-axis

# Create the viewer for rendering
viewer = mujoco.viewer.Viewer(sim)

# Simulate the system for a number of steps
for _ in range(1000):
    sim.step()  # Step the simulation forward
    viewer.render()  # Render the simulation in the viewer
