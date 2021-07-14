''' 
 File discription:
 Features and functions:


'''
from scipy.ndimage import grey_closing, grey_opening
import numpy as np
from skimage import measure


def segment_markers(image: np.ndarray, clip_thres=-600,
                    morph_size: tuple = (3, 3, 3),
                    binary_thres_percentile=96,
                    sum_z_thres=0.5,
                    marker_area_limit: list = [1500.0, 4000.0],
                    marker_diameter_limit: list = [10, 50],
                    marker_diameter_diff_limit=10.0
                    ):
    ''' segment the CT markers in a CT reconstruction image,
        return a segmented image and props for the markers.
    '''

    #   binarize the image with a emprical binary threshold and remove small structures in background.
    clip_image = np.zeros_like(image)
    clip_image[image >= clip_thres] = 1.0
    open_image = grey_opening(clip_image, size=morph_size)
    #   create a 2d reverse scan table mask along the axial direction
    sum_z_image = np.sum(open_image, axis=0)/open_image.shape[0]
    sum_z_image = sum_z_image/np.max(sum_z_image)
    mask_z_reverse = np.ones_like(sum_z_image)
    mask_z_reverse[sum_z_image > sum_z_thres] = 0.0

    #   remove the table by multiple the mask, ndarray broadcast is used here to multiply 3D array with 2D array.
    seg_image = open_image * mask_z_reverse

    #   second opening to remove the noise in segmented image with same kernel.
    open_seg = grey_opening(seg_image, size=morph_size)

    labels = measure.label(open_seg)
    props = measure.regionprops(labels)
    mal = marker_area_limit
    mdl = marker_diameter_limit
    mddl = marker_diameter_diff_limit
    valid_props = [p for p in props if p.area > mal[0] and p.area < mal[1] and p.minor_axis_length >
                   mdl[0] and p.major_axis_length < mdl[1] and (p.major_axis_length-p.minor_axis_length) < mddl]
    
    # if more than 2 valid props, select the largest two as valid props because the noise are usually of small pieces. 
    if len(valid_props) > 2:
        valid_props = sorted(
            valid_props, key=lambda y: y.area, reverse=True)[:2]

    return open_seg, valid_props
