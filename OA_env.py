import gym
import numpy as np
from gym import spaces
import torch
import mujoco
from load_fk import FKNN
from gym_env import MuJoCoEnv

class OA_env(gym.Env):
    def __init__(
        self,
        model_path="universal_robots_ur5e/scene.xml",
        max_episode_steps=200,
        phase_reward_weights=None,
        fk_weights_path="fk_nn_model.pth",
        device="cpu"
    ):
        super(OA_env, self).__init__()
        self.model_path = model_path
        self.max_episode_steps = max_episode_steps
        self.phase_reward_weights = phase_reward_weights or {"pickup": 1.0, "place": 1.0}
        self.device = device
        self.done = False
        self.mjc_env = MuJoCoEnv(self.model_path)
        self.fk_model = FKNN().to(self.device)
        if fk_weights_path is not None:
            self.fk_model.load_state_dict(torch.load(fk_weights_path, map_location=device))
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
        
        # Re-instantiate or define a dedicated reset in MuJoCoEnv
        self.mjc_env.__init__(self.model_path)
        self.steps = 0
        self.current_phase = 0
        return self._compute_observation()

    def step(self, action):
        self.steps += 1

        # Example discrete action interpretation:
        #   Even index => increment a joint
        #   Odd index  => decrement a joint
        #   joint_idx  = action // 2
        if action % 2 == 0:
            joint_idx = action // 2
            delta = 0.02
        else:
            joint_idx = action // 2
            delta = -0.02

        current_qpos = self.mjc_env.data.qpos[joint_idx]
        self.mjc_env.data.qpos[joint_idx] = current_qpos + delta

        self.mjc_env.log_contact_forces()
        
        obs = self._compute_observation()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)
        info = {}

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
        # Input to the FK
        # with torch.no_grad():
        #     qpos_robot_ip = qpos_robot[:6]
        #     stack = np.hstack([np.sin(qpos_robot_ip), np.cos(qpos_robot_ip)])
        #     inp = torch.FloatTensor(stack).unsqueeze(0).to(self.device)
        #     eef_out = self.fk_model(inp).squeeze(0).cpu().numpy()

        eef_body_id = mujoco.mj_name2id(self.mjc_env.model, mujoco.mjtObj.mjOBJ_BODY, "4boxes")        
        eef_pos = self.mjc_env.data.xpos[eef_body_id]
        eef_quat = self.mjc_env.data.xquat[eef_body_id]
        eef_out = np.concatenate((eef_pos, eef_quat), axis=0) 
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
        print(obs.shape)
        return obs

    def _compute_reward(self, obs):
        """
        Two-phase reward that includes:
        - Phase 0: distance-based shaping for approaching slab + correct contact (geom30 <-> geom31)
        - Phase 1: distance-based shaping to stand + specific 2-contact-per-geom 
        """
        reward = 0.0

        eef_x, eef_y, eef_z = obs[14], obs[15], obs[16]

        slab_x, slab_y, slab_z = obs[7], obs[8], obs[9]

        stand_x, stand_y = 2.0, 1.0

        contacts = self._get_contact_details()
        if self.current_phase == 0:
            dist_eef_slab = np.sqrt((eef_x - slab_x)**2 + (eef_y - slab_y)**2 + (eef_z - slab_z)**2)
            reward -= dist_eef_slab
            eef_slab_contact = False
            undesired_contacts = 0
            for c in contacts:
                g1, g2 = c["geom1"], c["geom2"]
                if {g1, g2} == {30, 31}:
                    eef_slab_contact = True
                else:
                    undesired_contacts += 1

            reward -= 5.0 * undesired_contacts

            if eef_slab_contact or dist_eef_slab < 0.05:
                reward += 10.0
                self.current_phase = 1

        else:
            dist_slab_stand = np.sqrt((slab_x - stand_x)**2 + (slab_y - stand_y)**2)
            reward -= dist_slab_stand

            positions_34 = []
            positions_36 = []
            big_force_count = 0
            undesired_contacts = 0
            
            for c in contacts:
                g1, g2 = c["geom1"], c["geom2"]
                force_6d = c["force"]
                pos_3d = c["pos"]    
                force_mag = np.linalg.norm(force_6d)

                if {g1, g2} == {30, 34}:
                    positions_34.append(pos_3d)
                    if force_mag > 50.0:
                        big_force_count += 1
                elif {g1, g2} == {30, 36}:
                    positions_36.append(pos_3d)
                    if force_mag > 50.0:
                        big_force_count += 1
                else:
                    undesired_contacts += 1
                    if force_mag > 50.0:
                        big_force_count += 1

            reward -= 5.0 * undesired_contacts

            reward -= 2.0 * big_force_count

            reward += min(len(positions_34), 2) * 1.0
            reward += min(len(positions_36), 2) * 1.0
            count_pairs_34 = self._count_pairs_with_y_spacing(positions_34, target_spacing=0.5, tol=0.05)
            count_pairs_36 = self._count_pairs_with_y_spacing(positions_36, target_spacing=0.5, tol=0.05)

            reward += 3.0 * count_pairs_34
            reward += 3.0 * count_pairs_36

            if len(positions_34) >= 2 and len(positions_36) >= 2:
                self.done = True
                if (count_pairs_34 > 0) and (count_pairs_36 > 0) and (big_force_count == 0) and (undesired_contacts == 0):
                    reward += 10.0
        return reward

    def _check_done(self, obs):
        if self.done or self.steps>=self.max_episode_steps:
            return True
        return False
    def _get_contact_details(self):
        """
        Returns a list of dictionaries, each describing one contact:
        {
            "geom1": <int>,
            "geom2": <int>,
            "force": np.array(6),  # the 6D contact force
            "pos": np.array(3),    # contact position in world coords
        }
        """
        contacts_info = []
        for i in range(self.mjc_env.data.ncon):
            c = self.mjc_env.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.mjc_env.model, self.mjc_env.data, i, force)

            contact_dict = {
                "geom1": c.geom1,
                "geom2": c.geom2,
                "force": force,
                "pos": np.array(c.pos), 
            }
            contacts_info.append(contact_dict)
        return contacts_info

    def render(self, mode='human'):
        self.mjc_env.render()
