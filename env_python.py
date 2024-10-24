import mujoco
import glfw
import numpy as np
import time

last_x, last_y = 0.0, 0.0
is_rotating = False
is_panning = False
is_zooming = False

cam_azimuth_speed = 0.2
cam_elevation_speed = 0.2
zoom_speed = 0.05
pan_speed = 0.01

ee_position = np.array([0.0, 0.0, 0.0])
ee_speed = 0.01 
min_z = 0.1 
max_z = 1.5 
x_limits = (-1.0, 1.0)  
y_limits = (-1.0, 1.0)
gripper_state = 0  

model = mujoco.MjModel.from_xml_path("ur10e/ur10e.xml")
data = mujoco.MjData(model)

if not glfw.init():
    raise Exception("Failed to initialize GLFW")

window = glfw.create_window(800, 600, "MuJoCo Simulation", None, None)
glfw.make_context_current(window)

scene = mujoco.MjvScene(model, maxgeom=1000)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
gripper_state:bool = False
cam.azimuth = 45
cam.elevation = -20
cam.distance = 5
cam.lookat = np.array([0.0, 0.0, 0.0])

gripper_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_geom")
glass_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "glass_panel")

def get_end_effector_position():
    wrist_3_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    return data.site_xpos[wrist_3_site_id].copy()



def toggle_gripper(a):
    gripper_state = not(a)
    data.ctrl[-1] = gripper_state
def is_gripper_touching_glass():
    for i in range(data.ncon):
        contact = data.contact[i]
        if (contact.geom1 == gripper_geom_id and contact.geom2 == glass_geom_id) or \
           (contact.geom1 == glass_geom_id and contact.geom2 == gripper_geom_id):
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, contact_force)
            return True
    return False

def move_end_effector(target_pos, current_pos):
    mujoco.mj_kinematics(model, data)
    wrist_3_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    displacement = target_pos - current_pos

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, wrist_3_site_id)

    # Damped pseudoinverse solution to avoid overshooting
    lambda_ = 1e-4
    delta_qpos = np.linalg.pinv(jacp.T @ jacp + lambda_ * np.eye(jacp.shape[1])) @ jacp.T @ displacement
    
    # Joint position limits for clipping
    qpos_min = model.jnt_range[:6, 0]
    qpos_max = model.jnt_range[:6, 1]

    # Update joint positions with a small step
    data.qpos[:6] = np.clip(data.qpos[:6] + delta_qpos[:6], qpos_min, qpos_max)
    print(data.qpos)
    mujoco.mj_forward(model, data)
def key_callback(window, key, scancode, action, mods):
    ee_position = get_end_effector_position()
    if action == glfw.PRESS or action == glfw.REPEAT:
        if key == glfw.KEY_LEFT:
            ee_position[1] = max(ee_position[1] - ee_speed, y_limits[0]) 
        elif key == glfw.KEY_RIGHT:
            ee_position[1] = min(ee_position[1] + ee_speed, y_limits[1]) 
        elif key == glfw.KEY_UP:
            ee_position[0] = min(ee_position[0] + ee_speed, x_limits[1])
        elif key == glfw.KEY_DOWN:
            ee_position[0] = max(ee_position[0] - ee_speed, x_limits[0])
        elif key == glfw.KEY_M:
            ee_position[2] = min(ee_position[2] + ee_speed, max_z)
        elif key == glfw.KEY_N:
            ee_position[2] = max(ee_position[2] - ee_speed, min_z)
        elif key == glfw.KEY_G:
            if is_gripper_touching_glass():
                toggle_gripper(gripper_state)
        else:
            return
        move_end_effector(ee_position, get_end_effector_position())

def mouse_button_callback(window, button, action, mods):
    global is_rotating, is_panning, is_zooming
    if button == glfw.MOUSE_BUTTON_LEFT:
        is_rotating = action == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        is_panning = action == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        is_zooming = action == glfw.PRESS

def generate_target_position():
    target_position = np.round(np.random.uniform(-2.0, 2.0, 2), 3)
    
    third_value = np.round(np.random.uniform(0.001, 2.0), 3)  # Ensure it is strictly > 0
    
    target_position = np.append(target_position, third_value)
    
    return target_position
def cursor_position_callback(window, xpos, ypos):
    global last_x, last_y, cam, is_rotating, is_panning, is_zooming
    
    dx, dy = xpos - last_x, ypos - last_y
    last_x, last_y = xpos, ypos

    if is_rotating:
        cam.azimuth += dx * cam_azimuth_speed
        cam.elevation = np.clip(cam.elevation - dy * cam_elevation_speed, -89.9, 89.9)
    
    elif is_panning:
        cam.lookat[0] -= dx * pan_speed
        cam.lookat[1] += dy * pan_speed
    
    elif is_zooming:
        cam.distance = max(0.1, cam.distance - dy * zoom_speed)

def scroll_callback(window, xoffset, yoffset):
    global cam
    cam.distance = max(0.1, cam.distance - yoffset * zoom_speed)
def new_target_position(current_pos, target_pos, epsilon=0.01, convergence_threshold=1e-5):
    new_target_pos = current_pos.copy()
    
    for i in range(3):
        if(i==0):
            print(f"Axis {i}")
        displacement = target_pos[i] - current_pos[i]
        if(i==0):
            print('displacement', displacement)

        if abs(displacement) < convergence_threshold:
            new_target_pos[i] = target_pos[i]  # Lock the position to the target if within threshold
            if(i==0):
                print(f"Axis {i} converged. Setting to target.")
        else:
            adaptive_epsilon = epsilon
            if(displacement <= epsilon):
                adaptive_epsilon = displacement
            # adaptive_epsilon = min(epsilon, abs(displacement / 10))  # Divide displacement by 10 for finer steps
            # adaptive_epsilon = max(adaptive_epsilon, convergence_threshold)  # Ensure epsilon doesn't become too small
            if(i==0):
                print('adaptive epsilon', adaptive_epsilon)
            step = np.clip(displacement, -adaptive_epsilon, adaptive_epsilon)
            if(i==0):
                print('step', step)
            new_target_pos[i] += step

    return new_target_pos
geom_name = "glass_panel"
geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
target_pos = np.array([0.8,1,0.6])
glfw.set_key_callback(window, key_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_position_callback)
glfw.set_scroll_callback(window, scroll_callback)
iterate = 0
while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)
    ee_position = get_end_effector_position()
    new_target_pos = new_target_position(ee_position, target_pos)
    print(target_pos-ee_position)
    move_end_effector(new_target_pos, ee_position)
    width, height = glfw.get_framebuffer_size(window)
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), scene, context)
    
    glfw.swap_buffers(window)
    
    glfw.poll_events()

    time.sleep(0.01)

glfw.terminate()
