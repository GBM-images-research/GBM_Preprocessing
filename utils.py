import os
import ants


class Modal:
    def __init__(self, output_path, document):
        self.path = os.path.join(output_path, document)
        self.ex = ants.image_read(self.path, reorient='IAL')
        
    def registration(self, template):
        self.transformation = ants.registration(
                fixed=template,
                moving=self.ex, 
                type_of_transform='SyN',
                verbose=True
            )
        self.reg = self.transformation['warpedmovout']

    def mask_image(self, brain_mask):
        self.masked = ants.mask_image(self.reg, brain_mask)

