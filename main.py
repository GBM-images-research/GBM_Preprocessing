import dicom2nifti
import os
import numpy as np
import ants
import matplotlib.pyplot as plt
import nibabel as nb

from glob import glob
from utils import *
from antspynet.utilities import brain_extraction
from scipy.ndimage import shift
from scipy.signal import correlate

def save_nifti_files(global_path, output_path):
    for i, modality in enumerate(glob(global_path)): #we save the nifti files 
        document = ''
        if i == 0 : 
            document = 't1c_test.nii'
        elif i == 1 : 
            document = 'flair_test.nii'
        elif i == 2 :
            document = 't2_test'
        else :
            document = 't1_test'
        
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

if __name__ == "__main__":
    # LOAD PATH FILES
    #MRI example dataset
    global_path = "C:\\Users\\Pablo\\Desktop\\PF\\GUI\\Imagenes\\TCGA-GBM\\TCGA-06-5413\\06-17-2008-18002\\*" 
    output_path = "C:\\Users\\Pablo\\Desktop\\PF\\GUI\\Imagenes\\TCGA-GBM\\NIFTI_FILES"
    #atlas dataset
    atlas_path_t1 = "C:\\Users\\Pablo\\Desktop\\PF\\GUI\\Imagenes\\sri24_spm8\\templates"

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
    atlas.set_origin((239,-239,0))
    brats_flag = True
    matrix = t1.coregistration(atlas,'Similarity', brats_flag)

    #Apply T1-Atlas transformation matrix to t1c, t2, flair
    t1c.apply_transformation(atlas,matrix)
    t2.apply_transformation(atlas,matrix)
    flair.apply_transformation(atlas,matrix)

    # T1 Brain extraction 
    prob_brain_mask = brain_extraction(t1.reg, modality="t1", verbose=True,)

    # GET T1 MASK
    brain_mask_t1 = ants.get_mask(prob_brain_mask, low_thresh=0.8)
    masked_t1 = ants.mask_image(t1.reg, brain_mask_t1)
    masked_t1.set_origin((239,-239,0))

    # Now we have the t1 mask, we do the same for t1c, t2 and flair
    t1c.mask_image(brain_mask_t1)
    t2.mask_image(brain_mask_t1)
    flair.mask_image(brain_mask_t1)

    # Plot

    ants.plot(masked_t1, figsize=1, axis=2)
    print('masked t1:', masked_t1)

    ants.plot(t1c.masked, figsize=1, axis=2)
    print('masked t1c:', t1c.masked)

    ants.plot(t2.masked, figsize=1, axis=2)
    print('masked t2:', t2.masked)

    ants.plot(flair.masked, figsize=1, axis=2)
    print('masked flair:', flair.masked)

    # Import control image

    path_control = "C:\\Users\\Pablo\\Desktop\\PF\\GUI\\Imagenes\\Pre-operative_TCGA_GBM_NIfTI_and_Segmentations\\TCGA-06-5413\\TCGA-06-5413_2008.06.17_t1.nii.gz"
    control = ants.image_read(path_control, reorient='IAL')
    
    # Otsu binarization
    bm1 = ants.segmentation.otsu.otsu_segmentation(masked_t1, 1, mask=None)
    bcontrol = ants.segmentation.otsu.otsu_segmentation(control, 1, mask=None) 

    # Max correlation
    bm1_shifted = max_correlation(bcontrol.numpy(),bm1.numpy())

    print("Propia:", bm1_shifted.shape)
    #print("Propia:", bm1.numpy().shape)
    print("Control:", bcontrol.numpy().shape)

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(bm1_shifted[:][:][20], cmap='gray')
    #axs[0].imshow(bm1.numpy()[:][:][20], cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('T1 binarizada')
    axs[1].imshow(bcontrol.numpy()[:][:][20], cmap='gray')
    axs[1].axis('off')
    axs[1].set_title('Control binarizada')
    plt.show()

    # Dice
    print(glob(global_path))
    resultado_dice = dice_coefficient(bm1_shifted, bcontrol.numpy())
    #resultado_dice = dice_coefficient(bm1.numpy(), bcontrol.numpy())
    print(f"Dice T1: {resultado_dice}")

    # Save shifted image as nii file
    #ni_img = nb.Nifti1Image(bm1_shifted, affine=np.eye(4))
    #nb.save(ni_img, os.path.join(output_path, "shifted"))

