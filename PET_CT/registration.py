''' 
 File discription: automatically segment the CT registration images and compute the RT matrix from PET to CT
 Features and functions:


'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import grey_closing, grey_opening
import numpy as np
from skimage import measure
import tqdm as tqdm
import pandas as pd


class PetCtAnnotation:
    def __init__(self, load: str, ct_id: str, pet_id: str, ct_point: np.ndarray = None, pet_point: np.ndarray = None):
        self.load = load
        self.ct_id = ct_id
        self.pet_id = pet_id
        self.ct_point = ct_point
        self.pet_point = pet_point

    def print_self(self):
        print(self.load, self.ct_id, self.pet_id,
              self.ct_point, self.pet_point)


def read_ct_image(file_dir):
    cfg_file = file_dir+'/config.json'
#     print(cfg_file)
    with open(cfg_file) as f:
        cfg = json.load(f)
#     pprint.pprint(cfg)
    image_meta = cfg['acct_image']['image_meta']
    ct_shape = np.array(
        [image_meta['shape']['x'], image_meta['shape']['y'], image_meta['shape']['z']])
#     print(ct_shape)
    image3d_id = cfg['image3d_id']
    ct_image = np.fromfile(
        file_dir + '/' + cfg['acct_image']['raw_path'], dtype=np.int16, ).reshape(ct_shape[::-1])
    return image3d_id, ct_image, image_meta


class CtImage:
    def __init__(self, image3d_id, data, meta):
        self.image3d_id = image3d_id
        self.data = data
        self.parse_meta(meta)

    def parse_meta(self, meta_dict):
        aff_dict = meta_dict['affine']
        self.affmat = np.array([aff_dict[ele]
                               for ele in aff_dict]).reshape(4, 4)
        shape_dict = meta_dict['shape']
        self.shape = np.squeeze(
            np.array([shape_dict[ele] for ele in shape_dict]).reshape((1, 3)))


def show3(image, pos: list, vs: list = [None, None], cmap=plt.cm.gray):
    plt.subplot(131)
    plt.imshow(image[pos[0], :, :], vmin=vs[0], vmax=vs[1], cmap=cmap)
    plt.subplot(132)
    plt.imshow(image[:, pos[1], :], vmin=vs[0], vmax=vs[1], cmap=cmap)
    plt.subplot(133)
    plt.imshow(image[:, :, pos[2]], vmin=vs[0], vmax=vs[1], cmap=cmap)


def segment_markers(image: np.ndarray, clip_thres=-600,
                    morph_size: tuple = (3, 3, 3),
                    binary_thres_percentile=96,
                    sum_z_thres=0.5,
                    marker_area_limit: list = [1500.0, 4000.0],
                    marker_diameter_limit: list = [10, 50],
                    marker_diameter_diff_limit=10.0
                    ):
    ''' 

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
#     if len(valid_props) > 2:
    valid_props = sorted(valid_props, key=lambda y: y.area, reverse=True)[:2]

    return open_seg, valid_props


def to_bed_coordinate(affmat, pt, shape):
    pt = pt[::-1]/shape-0.5
    tpt = affmat[:3, :3]@pt + affmat[:3, 3]
#     print(pt, tpt)
    return tpt


def create_auto_pet_dict(auto_pet_labels):
    apl = auto_pet_labels
    auto_pet_dict_large = dict()
    auto_pet_dict_small = dict()
    pet_label_list = [[pid, tag, [x, y, z]] for pid, tag, x, y, z in zip(
        apl['image3d_id'], apl['tag'], apl['x'], apl['y'], apl['z'])]
    for item in pet_label_list:
        pid, tag, pt = item
        if tag == 'py_auto_big':
            auto_pet_dict_large[pid] = np.array(pt)
        elif tag == 'py_auto_sml':
            auto_pet_dict_small[pid] = np.array(pt)
    auto_pet_dict = dict()

    for item in pet_label_list:
        pid, tag, pt = item
        auto_pet_dict[pid] = [
            auto_pet_dict_large[pid], auto_pet_dict_small[pid]]
    return auto_pet_dict


def create_pet_ct_label_list(manu_labels, auto_pet_dict, auto_ct_dict):
    label_list = []
    pet_ct_label_set = set([(des, pid, cid) for des, pid, cid in zip(
        manu_labels['description'], manu_labels['pet_image3d_id'], manu_labels['image3d_id'])])
    for item in pet_ct_label_set:
        des, pid, cid = item
        for ct_pos in auto_ct_dict[pid]:
            for pet_pos in auto_pet_dict[pid]:
                diff = ct_pos - pet_pos
                if np.max(np.abs(diff)) < 10:
                    label_list.append(PetCtAnnotation(
                        des, cid, pid, ct_pos, pet_pos))
    return label_list


def compute_Rt_error(rt_mat, pet_pts, ct_pts):
    r0, t0 = rt_mat
#     err = ct_pts - (r0 @ pet_pts + t0)
    err = ct_pts - (pet_pts + t0)
    return err
