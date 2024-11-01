import mujoco
import numpy as np
class ContactForce:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.gripping = False

    def contact_pts(self,slab_geom_id, arm_geom_id):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if(contact.geom1 in [slab_geom_id] or contact.geom2 in [slab_geom_id]):
                self.gripping = True
                force_contact_frame = np.zeros((6, 1))
                mujoco.mj_contactForce(self.model, self.data, i, force_contact_frame)

                normal_force = force_contact_frame[0][0]
                tangential_force_1 = force_contact_frame[1][0]
                tangential_force_2 = force_contact_frame[2][0]
                print(force_contact_frame)
                print(f"Contact {i} between geom {contact.geom1} and geom {contact.geom2}:")
                print(f"  Normal force: {normal_force}")
                print(f"  Tangential force 1: {tangential_force_1}")
                print(f"  Tangential force 2: {tangential_force_2}")                
                return 
