import dicom2nifti
import os
import matplotlib.pyplot as plt
import ants
import SimpleITK as sitk

from glob import glob
from utils import *
from antspynet.utilities import brain_extraction


def save_nifti_files(global_path, output_path):
    for i, modality in enumerate(glob(global_path)): #we save the nifti files 
        document = ''
        if i == 0 : 
            document = 't2_test.nii'
        elif i == 1 : 
            document = 't1c_test.nii'
        elif i == 2 :
            document = 't1_test'
        else :
            document = 'flair_test'
        
        dicom2nifti.dicom_series_to_nifti(modality, os.path.join(output_path, document))

if __name__ == "__main__":
    # LOAD PATH FILES
    global_path = '/Users/Maxy/Desktop/GBM/Herramienta CAD/TCGA-GBM/TCGA-02-0003/06-08-1997-MRI BRAIN WWO CONTRAMR-81239/*' 
    output_path = '/Users/Maxy/Desktop/GBM/Herramienta CAD/TCGA-GBM/TCGA-02-0003/NIFTI_FILES'

    save_nifti_files(global_path, output_path)

    t1 = Preprocessing(output_path, 't1_test.nii')
    t1c = Preprocessing(output_path, 't1c_test.nii')
    t2 = Preprocessing(output_path, 't2_test.nii')
    flair = Preprocessing(output_path, 'flair_test.nii')

    # CO REGISTRATION
    template = t1.temp

    t1c.registration(template)
    t2.registration(template)
    flair.registration(template)

    # T1 Brain extraction 
    prob_brain_mask = brain_extraction(template, modality="t1", verbose=True,)

    # GET T1 MASK
    brain_mask_t1 = ants.get_mask(prob_brain_mask, low_thresh=0.5)
    masked_t1 = ants.mask_image(template, brain_mask_t1)

    # Now we have the t1 mask, we do the same for t1c, t2 and flair
    t1c.mask_image(brain_mask_t1)
    t2.mask_image(brain_mask_t1)
    flair.mask_image(brain_mask_t1)
