import mujoco
import glfw
import numpy as np
import time


class MujocoViewerWithSliders:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.is_alive = True

        # Initialize GLFW
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        self.window = glfw.create_window(1200, 900, "MuJoCo Viewer with Sliders", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # MuJoCo visualization elements
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Joint slider setup
        self.joint_values = np.zeros(self.model.nq)
        self.slider_positions = [(50, 50 + i * 30) for i in range(self.model.nq)]
        self.slider_width = 200
        self.slider_height = 15

        # Interaction variables
        self.active_slider = None
        self.viewport = mujoco.MjrRect(0, 0, 1200, 900)

        # Mouse callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)

    def _draw_slider(self, index, xpos, ypos, width, value):
        """Draw a compact slider with labels."""
        label = f"Joint {index + 1}: {value:.2f} rad"

        # Draw the slider background
        mujoco.mjr_text(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            label,
            xpos - 10,
            ypos,
            [0.2, 0.2, 0.2, 1.0],
            self.ctx,
        )

        # Slider bar
        bar_x = xpos + len(label) * 8
        bar_width = width
        fill_width = int((value + 1) * 0.5 * width)  # Map [-1, 1] to [0, width]

        mujoco.mjr_rectangle(
            mujoco.MjrRect(bar_x, ypos - self.slider_height // 2, bar_x + bar_width, ypos + self.slider_height // 2),
            self.ctx,
            [0.6, 0.6, 0.6, 1.0],
        )
        mujoco.mjr_rectangle(
            mujoco.MjrRect(bar_x, ypos - self.slider_height // 2, bar_x + fill_width, ypos + self.slider_height // 2),
            self.ctx,
            [0.2, 0.7, 0.2, 1.0],
        )

    def _mouse_button_callback(self, window, button, action, mods):
        """Handle mouse clicks for sliders."""
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)
            for i, (sx, sy) in enumerate(self.slider_positions):
                if sx <= xpos <= sx + self.slider_width and sy - 10 <= ypos <= sy + 10:
                    self.active_slider = i
                    break
        elif action == glfw.RELEASE:
            self.active_slider = None

    def _cursor_pos_callback(self, window, xpos, ypos):
        """Handle slider value updates."""
        if self.active_slider is not None:
            slider_x, slider_y = self.slider_positions[self.active_slider]
            value = (xpos - slider_x) / self.slider_width * 2 - 1  # Normalize to [-1, 1]
            value = max(min(value, 1.0), -1.0)  # Clamp to [-1, 1]
            self.joint_values[self.active_slider] = value

    def render(self):
        """Render the MuJoCo environment and sliders."""
        if glfw.window_should_close(self.window):
            self.close()
            return

        width, height = glfw.get_framebuffer_size(self.window)
        self.viewport.width, self.viewport.height = width, height

        # Update the MuJoCo scene
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)

        # Draw the sliders
        for i, (x, y) in enumerate(self.slider_positions):
            self._draw_slider(i, x, y, self.slider_width, self.joint_values[i])

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        """Close the viewer."""
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()


if __name__ == "__main__":
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)

    # Create the viewer
    viewer = MujocoViewerWithSliders(model, data)

    while viewer.is_alive:
        # Update joint positions based on slider values
        for i in range(model.nq):
            data.qpos[i] = viewer.joint_values[i]  # Set joint positions
        mujoco.mj_step(model, data)  # Step the simulation
        viewer.render()
