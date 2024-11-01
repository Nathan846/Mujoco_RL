import mujoco
import numpy as np
class ContactForce:
    def __init__(self, model, data,slab_geom_id, arm_geom_id):
        self.model = model
        self.data = data
        self.gripping = False
        self.slab_geom_id = slab_geom_id
        self.arm_geom_id = arm_geom_id
        self.suction_force = 50
    def contact_pts(self,):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if(contact.geom1 in [self.slab_geom_id, self.arm_geom_id] or contact.geom2 in [self.slab_geom_id,self.arm_geom_id]):
                self.gripping = True
                force_contact_frame = np.zeros((6, 1))
                mujoco.mj_contactForce(self.model, self.data, i, force_contact_frame)

                normal_force = force_contact_frame[0][0]
                tangential_force_1 = force_contact_frame[1][0]
                tangential_force_2 = force_contact_frame[2][0]
                # print(f"Contact {i} between geom {contact.geom1} and geom {contact.geom2}:")
                # print(f"  Normal force: {normal_force}")
                # print(f"  Tangential force 1: {tangential_force_1}")
                # print(f"  Tangential force 2: {tangential_force_2}")                
                return 
    def gripper_force(self):
        if(self.gripping):
            vacuum_pos = self.data.geom_xpos[self.arm_geom_id]
            slab_pos = self.data.geom_xpos[self.slab_geom_id]
            direction = slab_pos - vacuum_pos
            distance = np.linalg.norm(direction)
            
            if distance < 1e-2:
                print('directionss')
                direction /= distance  
                force = self.suction_force * direction
                self.data.xfrc_applied[self.slab_geom_id, :3] = -force  