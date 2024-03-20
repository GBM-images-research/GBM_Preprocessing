import os
import ants
import dicom2nifti
from glob import glob
import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate


class Preprocessing:
    def __init__(self, output_path, document):
        self.path = os.path.join(output_path, document)
        self.temp = ants.image_read(self.path, reorient='IAL')

    def coregistration(self, template, transform, brats_flag):
        self.transformation = ants.registration(
                fixed=template,
                moving=self.temp, 
                type_of_transform=transform,
                verbose=True
            )
        self.reg = self.transformation['warpedmovout']
        if(brats_flag == True):
            return self.transformation

    
    def apply_transformation(self,atlas,matrix):
        transform_paths = [matrix['fwdtransforms'][0]]
        self.transformed_image = ants.apply_transforms(
                fixed=atlas,  # must be atlas
                moving=self.reg,
                transformlist=transform_paths,
                interpolator='linear',  
                imagetype=0,  
                verbose=True
            )
        self.res = self.transformed_image
    def mask_image(self, brain_mask):
        self.masked = ants.mask_image(self.res, brain_mask)


def save_nifti_files(global_path, output_path):
    for i, modality in enumerate(glob(global_path)): #we save the nifti files 
        document = ''
        if i == 0 : 
            document = 't2_test'
        elif i == 1 : 
            document = 't1_test.nii'
        elif i == 2 :
            document = 'flair_test.nii'
        else :
            document = 't1c_test'
        
        dicom2nifti.dicom_series_to_nifti(modality, os.path.join(output_path, document))

def dice_coefficient(volume1, volume2):
    intersection = np.sum(volume1 * volume2)
    total_voxels = np.sum(volume1) + np.sum(volume2)

    dice = (2.0 * intersection) / total_voxels

    return dice

def max_correlation(control, imagen):

    # Calculate 3D cross-correlation 
    correlation = correlate(control, imagen, mode="same", method="fft")

    # Max correlation coords
    coords = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Calculate shift
    shift_values = np.array(coords) - np.array(control.shape) // 2

    # Apply shift
    im_shifted = shift(imagen, shift_values, mode='constant', cval=0)

    return im_shifted