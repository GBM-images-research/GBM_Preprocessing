import os
import ants


class Preprocessing:
    def __init__(self, output_path, document):
        self.path = os.path.join(output_path, document)
        self.temp = ants.image_read(self.path, reorient='IAL')
        
    def registration(self, template):
        self.transformation = ants.registration(
                fixed=template,
                moving=self.temp, 
                type_of_transform='SyN',
                verbose=True
            )
        self.reg = self.transformation['warpedmovout']

    def mask_image(self, brain_mask):
        self.masked = ants.mask_image(self.reg, brain_mask)

