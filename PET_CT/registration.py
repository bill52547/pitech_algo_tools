''' 
 File discription:
 Features and functions:


''' 
from scipy.ndimage import grey_closing, grey_opening
import numpy as np
from skimage import measure

def segment_markers(image:np.ndarray, clip_thres = -600, 
                   morph_size:tuple = (3,3,3), 
                   binary_thres_percentile = 96,
                   table_sum_thres = 0.5,
                   minimal_marker_size = 256
                  ):
    ''' segment the CT markers in a CT reconstruction image,
        return a segmented image and props for the markers.
    '''

#   clip the background pixel to minimal intersity  
    clip_image = np.array(image)
    clip_image[image<clip_thres] = np.min(image)
#   remove small structures in background and binarize the image with a emprical binary threshold
    open_image = grey_opening(clip_image, size = morph_size)
    bin_image = np.zeros_like(open_image)
    threshold = np.percentile(open_image, binary_thres_percentile)
    bin_image[open_image>threshold] = 1.0 
#   create a 2d reverse scan table mask along the axial direction 
    sum_z_image = np.sum(bin_image, axis=0)/bin_image.shape[0]
    mask_z_reverse = np.ones_like(sum_z_image)
    mask_z_reverse[sum_z_image>table_sum_thres] = 0.0
#   remove the table by multiple the mask, ndarray broadcast is used here to multiply 3D array with 2D array.
    seg_image = bin_image * mask_z_reverse

#   second opening to remove the noise in segmented image with same kernel.
    open_seg = grey_opening(seg_image, size=morph_size)

#   
    labels = measure.label(seg_image)
    props = measure.regionprops(labels)
    valid_props  = [ p for p in props if p.area > minimal_marker_size ]

    return open_seg, valid_props 