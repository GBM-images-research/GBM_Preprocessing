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
    #MRI example dataset
    global_path = '/Users/Maxy/Desktop/GBM/Herramienta_CAD/TCGA-GBM/TCGA-02-0003/06-08-1997-MRI BRAIN WWO CONTRAMR-81239/*' 
    output_path = '/Users/Maxy/Desktop/GBM/Herramienta_CAD/TCGA-GBM/TCGA-02-0003/NIFTI_FILES'
    #atlas dataset
    atlas_path_t1 = '/Users/Maxy/Desktop/GBM/Herramienta_CAD/sri24_spm8/templates'

    save_nifti_files(global_path, output_path)

    t1 = Preprocessing(output_path, 't1_test.nii')
    t1c = Preprocessing(output_path, 't1c_test.nii')
    t2 = Preprocessing(output_path, 't2_test.nii')
    flair = Preprocessing(output_path, 'flair_test.nii')
    atlas_t1 = Preprocessing(atlas_path_t1,'T1.nii' )

    # CO REGISTRATION
    template = t1.temp
    brats_flag = False
    t1c.coregistration(template,'Similarity', brats_flag)
    t2.coregistration(template,'Similarity', brats_flag)
    flair.coregistration(template,'Similarity', brats_flag)

    # Native space transformation 
    atlas = atlas_t1.temp
    brats_flag = True
    matrix = t1.coregistration(atlas,'SyN', brats_flag)

    #Apply T1-Atlas transformation matrix to t1c, t2, flair
    t1c.apply_transformation(atlas,matrix)
    t2.apply_transformation(atlas,matrix)
    flair.apply_transformation(atlas,matrix)

    # T1 Brain extraction 
    prob_brain_mask = brain_extraction(t1.reg, modality="t1", verbose=True,)

    # GET T1 MASK
    brain_mask_t1 = ants.get_mask(prob_brain_mask, low_thresh=0.5)
    masked_t1 = ants.mask_image(t1.reg, brain_mask_t1)

    # Now we have the t1 mask, we do the same for t1c, t2 and flair
    t1c.mask_image(brain_mask_t1)
    t2.mask_image(brain_mask_t1)
    flair.mask_image(brain_mask_t1)

    ants.plot(masked_t1, figsize=1, axis=2)
    print('masked t1:', masked_t1)

    ants.plot(t1c.masked, figsize=1, axis=2)
    print('masked t1:', t1c.masked)

    ants.plot(t2.masked, figsize=1, axis=2)
    print('masked t1:', t2.masked)

    ants.plot(flair.masked, figsize=1, axis=2)
    print('masked t1:', flair.masked)
