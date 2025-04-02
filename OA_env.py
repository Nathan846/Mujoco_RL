"""
Defining env here, includi9ng reward functions, observation actions spaces etc
"""
import gym
import numpy as np
from gym import spaces
import torch
import mujoco
from load_fk import FKNN
from mujoco_env import MuJoCoEnv

class OA_env(gym.Env):
    def __init__(
        self,
        model_path="universal_robots_ur5e/scene.xml",
        max_episode_steps=50000,
        phase_reward_weights=None,
        fk_weights_path="fk_nn_model.pth",
        device="cpu",
        render = True
    ):
        super(OA_env, self).__init__()
        self.model_path = model_path
        self.max_episode_steps = max_episode_steps
        self.phase_reward_weights = phase_reward_weights or {"pickup": 1.0, "place": 1.0}
        self.device = device
        self.done = False
        self.render_var = True
        self.mjc_env = MuJoCoEnv(self.model_path)
        self.fk_model = FKNN().to(self.device)
        # if fk_weights_path is not None:
            # self.fk_model.load_state_dict(torch.load(fk_weights_path, map_location=device))
        self.fk_model.eval()

        # structure:
        #   - Robot joint angles: 7
        #   - Slab pose: 7 (3 position + 4 quaternion)
        #   - End-effector pose from FK: 7 (x, y, z, qw, qx, qy, qz)
        #   - contact force: 4*2*4(floats) = 32
        #       - 4 buckets, one representing (30,31), (31,34), (31,36) and everything else. And 2 points per bucket
        #         Per bucket - 4 floats representing force from 3 directoins and sum(MIGHT CHANGE)
        #      
        # => total 53 dims
        self.obs_dim = 53 
        high = np.inf * np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # -----------------------
        # action space
        # -----------------------
        # 7 joints, each can be incremented or decremented => 14 discrete actions
        self.num_joints = 7
        self.actions_per_joint = 2
        self.action_space = spaces.Discrete(self.num_joints * self.actions_per_joint)

        self.steps = 0
        self.current_phase = 0  # 0 => pick, 1 => place

    def reset(self):
        
        self.initial_slab_quat = np.array([1, 0, 0, 0])
        slab_joint_id = mujoco.mj_name2id(self.mjc_env.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.mjc_env.model.jnt_qposadr[slab_joint_id]
        slab_pos = [1.5, 0.5, 0.01]
        self.mjc_env.welded = False
        self.mjc_env.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 3] = slab_pos
        init_quat = [1]
        self.mjc_env.data.qpos[slab_qpos_start_idx + 3:slab_qpos_start_idx + 7] = [1,0,0,0]
        self.mjc_env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.mjc_env.model, self.mjc_env.data)
        for i in range(7):
            self.mjc_env.data.qpos[i] = 0.0
        self.done = False
        self.steps = 0
        self.current_phase = 0
        return self._compute_observation()

    def step(self, action):
        self.steps += 1
        if action is None:
            obs = self._compute_observation()
            reward = self._compute_reward(obs)
            done = self._check_done(obs)
            return obs, reward, done, {}
        joint_idx = action // 2
        direction = 1 if action % 2 == 0 else -1

        base_delta = 0.0004363326388885369
        delta = np.radians(0.025)*2 if joint_idx == 2 else base_delta
        delta *= direction

        current_qpos = self.mjc_env.data.qpos[joint_idx]
        self.mjc_env.data.qpos[joint_idx] = current_qpos + delta

        mujoco.mj_forward(self.mjc_env.model, self.mjc_env.data)
        obs = self._compute_observation()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)
        info = {}
        # self.render()
        return obs, reward, done, info

    def _compute_observation(self):
        """
        Observations contains:
        - 7 robot joint angles
        - 7 slab pose
        - 7 EEF pose from FK
        - Then contact data for 4 categories:
            (30,31), (31,34), (31,36), and everything else
        Each category stores up to 2 contact points (pos_x, pos_y, pos_z, force_mag).
        """

        qpos_robot = self.mjc_env.data.qpos[:7].copy()
        slab_joint_id = mujoco.mj_name2id(self.mjc_env.model, mujoco.mjtObj.mjOBJ_JOINT, "slab_free")
        slab_qpos_start_idx = self.mjc_env.model.jnt_qposadr[slab_joint_id]
        slab_pose = self.mjc_env.data.qpos[slab_qpos_start_idx:slab_qpos_start_idx + 7]
        # with torch.no_grad():
        #     qpos_robot_ip = qpos_robot[:6]
        #     stack = np.hstack([np.sin(qpos_robot_ip), np.cos(qpos_robot_ip)])
        #     inp = torch.FloatTensor(stack).unsqueeze(0).to(self.device)
        #     eef_out = self.fk_model(inp).squeeze(0).cpu().numpy()

        eef_body_id = mujoco.mj_name2id(self.mjc_env.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")        
        eef_pos = self.mjc_env.data.xpos[eef_body_id]
        eef_quat = self.mjc_env.data.xquat[eef_body_id]
        eef_out = np.concatenate((eef_pos, eef_quat), axis=0) 
        self.eef_out = eef_out
        self.slab_pose = slab_pose
        base_obs = np.concatenate([qpos_robot, slab_pose, eef_out])

        contacts_info = self._get_contact_details()

        bucket_30_31 = []
        bucket_31_34 = []
        bucket_31_36 = []
        bucket_others = []

        for c in contacts_info:
            g1, g2 = c["geom1"], c["geom2"]
            fvec = c["force"]
            pos  = c["pos"]
            pair_set = {g1, g2}

            if pair_set == {30, 31}:
                bucket_30_31.append(c)
            elif pair_set == {31, 34}:
                bucket_31_34.append(c)
            elif pair_set == {31, 36}:
                bucket_31_36.append(c)
            else:
                bucket_others.append(c)

        max_points_per_group = 2

        def bucket_to_array(bucket):
            arr = []
            for i in range(max_points_per_group):
                if i < len(bucket):
                    cdict = bucket[i]
                    px, py, pz = cdict["pos"]
                    force_6d = cdict["force"]
                    fmag = np.linalg.norm(force_6d)
                    arr += [px, py, pz, fmag]
                else:
                    arr += [0.0, 0.0, 0.0, 0.0]
            return arr

        arr_30_31 = bucket_to_array(bucket_30_31) 
        arr_31_34 = bucket_to_array(bucket_31_34)
        arr_31_36 = bucket_to_array(bucket_31_36)
        arr_others = bucket_to_array(bucket_others)

        contact_obs = arr_30_31 + arr_31_34 + arr_31_36 + arr_others
        contact_obs_arr = np.array(contact_obs, dtype=np.float32)

        obs = np.concatenate([base_obs, contact_obs_arr]).astype(np.float32)
        return obs

    def _compute_reward(self, obs):
        reward = 0.0

        eef_x, eef_y, eef_z = obs[14], obs[15], obs[16]
        slab_x, slab_y, slab_z = obs[7], obs[8], obs[9]
        eef_quat = obs[17:21]
        slab_quat = obs[10:14]
        slab_quat[3] = -0.5
        

        contacts = self._get_contact_details()

        if self.current_phase == 0:
            dist_eef_slab = np.sqrt((eef_x - slab_x)**2 + (eef_y - slab_y)**2 + (eef_z - slab_z)**2)
            reward -= dist_eef_slab

            dot_product = np.clip(np.dot(eef_quat, slab_quat), -1.0, 1.0)
            angle_diff = 2 * np.arccos(abs(dot_product))
            reward -= 5.0 * angle_diff

            # Contact check
            eef_slab_contact = False
            undesired_contacts = 0
            for c in contacts:
                g1, g2 = c["geom1"], c["geom2"]
                if {g1, g2} == {30, 31}:
                    eef_slab_contact = True
                else:
                    if({g1,g2} == {0,31} or {g1,g2}=={27,30}):
                        continue
                    if(g1==0 or (g1==27 and g2==30)):
                        continue
                    undesired_contacts += 1
            reward -= 5.0 * undesired_contacts

            if eef_slab_contact or dist_eef_slab < 0.05:
                reward += 10.0
                self.current_phase = 1
        else:
            reward = 0
            ideal_eef_pos = np.array([-0.11150263, -0.133614, 0.3382516])
            ideal_eef_quat = np.array([0.89702304, 0.09909241, 0.37242599, 0.21640066])

            eef_pos = np.array([eef_x, eef_y, eef_z])
            eef_quat = eef_quat / np.linalg.norm(eef_quat + 1e-8)
            pos_dist = np.linalg.norm(eef_pos - ideal_eef_pos)
            pos_reward = (1.5 - pos_dist) * 5.0
            reward += pos_reward
            
            quat_dot = np.clip(np.dot(eef_quat, ideal_eef_quat), -1.0, 1.0)
            angle_diff = 2 * np.arccos(abs(quat_dot))
            quat_reward = (1.0 - angle_diff) *5
            reward += quat_reward

            # print(f"EEF-ideal dist: {pos_dist:.3f}, angle_diff: {angle_diff:.3f} → pos_r: {pos_reward:.2f}, quat_r: {quat_reward:.2f}")
            positions_36 = []
            positions_39 = []
            forces_36 = []
            forces_39 = []
            undesired_contacts = 0

            for c in contacts:
                g1, g2 = c["geom1"], c["geom2"]
                force_6d = c["force"]
                pos_3d = c["pos"]
                force_mag = np.linalg.norm(force_6d)

                if {g1, g2} == {31, 36}:
                    positions_36.append(pos_3d)
                    forces_36.append(force_mag)

                elif {g1, g2} == {31, 39}:
                    positions_39.append(pos_3d)
                    forces_39.append(force_mag)

                # Allow other specified contacts
                elif {g1, g2} in [{27, 30}, {30, 31}]:
                    continue
                elif {g1,g2} == {30,37}:
                    continue
                else:
                    if(g1==30 and g2==37):
                        continue
                    undesired_contacts += 1
                    if force_mag > 60.0:
                        reward -= 5.0  # Heavily penalize high-magnitude undesired contacts
                    else:
                        reward -= 2.0
            
            reward -= 5.0 * undesired_contacts  # General penalty for undesired contacts

            # --- Positional Reward ---
            reward += min(len(positions_36), 2) * 1.0
            reward += min(len(positions_39), 2) * 1.0

            count_pairs_36 = self._count_pairs_with_y_spacing(positions_36, target_spacing=0.5, tol=0.05)
            count_pairs_39 = self._count_pairs_with_y_spacing(positions_39, target_spacing=0.5, tol=0.05)

            reward += 3.0 * count_pairs_36
            reward += 3.0 * count_pairs_39

            def force_score(forces, label):
                if len(forces) < 2:
                    return 0.0, False
                avg_force = np.mean(forces)
                if avg_force > 60.0:
                    penalty = -5.0 * (avg_force - 60.0)
                    print(f"[{label}] High avg force: {avg_force:.2f} → penalty {penalty:.2f}")
                    return penalty, False
                else:
                    reward_force = max(0.5, 2.0 - (avg_force / 60.0))  # reward drops off as avg_force increases
                    print(f"[{label}] Good avg force: {avg_force:.2f} → reward {reward_force:.2f}")
                    return reward_force, True
            
            force_r_36, valid_36 = force_score(forces_36, "Beam 36")
            force_r_39, valid_39 = force_score(forces_39, "Beam 39")

            reward += force_r_36 + force_r_39
            # print(len(positions_36), len(positions_39), valid_36, valid_39,'solid and good')
            if (len(positions_36) >= 2 and len(positions_39) >= 2 and 
                valid_36 and valid_39 and undesired_contacts == 0):

                self.done = True
                print("✔️ Placement successful!")
                reward += 10.0
        return reward
    def _count_pairs_with_y_spacing(self, positions, target_spacing=0.5, tol=0.05):
        count = 0
        n = len(positions)
        for i in range(n):
            for j in range(i+1, n):
                y_i = positions[i][1]
                y_j = positions[j][1]
                spacing = abs(y_i - y_j)
                if abs(spacing - target_spacing) <= tol:
                    count += 1
        return count

    def _check_done(self, obs):
        if self.done or self.steps>=self.max_episode_steps:
            return True
        return False
    def _get_contact_details(self):
        contacts_info = []
        for i in range(self.mjc_env.data.ncon):
            c = self.mjc_env.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.mjc_env.model, self.mjc_env.data, i, force)
            if c.geom1 == 30 and c.geom2 == 31:
                    self.mjc_env.welded = True
            contact_dict = {
                "geom1": c.geom1,
                "geom2": c.geom2,
                "force": force,
                "pos": np.array(c.pos), 
            }
            contacts_info.append(contact_dict)
        return contacts_info

    def render(self, mode='human'):
        if(self.render_var):    
            self.mjc_env.render()
