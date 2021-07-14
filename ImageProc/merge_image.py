import numpy as np
import os
import json
import numpy as np
import srfnef as nef
import json


def load_image(target_dir):
    img_file = os.path.join(target_dir, 'result_data.bin.hdf5')
    conf_file = os.path.join(target_dir, 'result_meta.json')
    with open(conf_file, 'r') as fin:
        conf = json.load(fin)
    img = nef.load(nef.Image, img_file)
    new_center = [conf['affine']['m14'], conf['affine']['m24'], conf['affine']['m34']]
    img = img.update(center = new_center)
    return img

def get_grid_z(img):
    nx, ny, nz = img.shape
    sx, sy, sz = np.round(img.size, decimals=2)
    cx, cy, cz = np.round(img.center, decimals=2)
    dx, dy, dz = np.round(img.unit_size, decimals=2)
    grid_x = np.linspace(cx - sx / 2 + 0.5 * dx, cx + sx / 2 - 0.5 * dx, nx)
    grid_y = np.linspace(cy - sy / 2 + 0.5 * dy, cy + sy / 2 - 0.5 * dy, ny)
    grid_z = np.linspace(cz - sz / 2 + 0.5 * dz, cz + sz / 2 - 0.5 * dz, nz)
    return grid_z

def merge_image(img_list):
    img = img_list[0]
    sx, sy, sz = img.size
    nx, ny, nz = img.shape
    dx, dy, dz = img.unit_size
    cx, cy, cz = img.center

    def get_weight_slice(img, z):
        range_z = img.center[2] - sz / 2 + dz / 2, img.center[2] + sz / 2 - dz / 2
        grid_z = get_grid_z(img)
        if z > range_z[1] or z < range_z[0]:
            return np.zeros((nx, ny), dtype = np.float32)
        for iz in range(nz - 1):
            if grid_z[iz] <= z < grid_z[iz + 1]:
                break
        w2, w1 = (z - grid_z[iz]) / dz, 1 - (z - grid_z[iz]) / dz
        return img.data[:,:,iz] * w1 + img.data[:,:,iz + 1] * w2


    def find_nearest_img(img_list, z):
        center_z_list = np.array([img.center[2] for img in img_list])
        return np.argmin(np.abs(center_z_list - z))

    def find_nearest_2_img(img_list, z):
        num_img = len(img_list)
        center_z_list = np.array([img.center[2] for img in img_list])
        if z < center_z_list[0]:
            return -1, 0, -1, center_z_list[0] - z
        if z >= center_z_list[-1]:
            return num_img - 1, -1, z - center_z_list[-1], -1
        for iz, z0 in enumerate(center_z_list):
            z1 = center_z_list[iz + 1]
            if z0 <= z < z1:
                break
        return iz, iz + 1, z - z0, z1 - z

    center_z_list = [img.center[2] for img in img_list]
    print(center_z_list)
    min_center_z = np.min(center_z_list)
    max_center_z = np.max(center_z_list)
    full_size_z = max_center_z - min_center_z + sz
    full_center_z = (max_center_z + min_center_z) / 2
    full_shape_z = np.ceil(full_center_z / dz).astype(np.int)
    full_img = nef.Image(np.zeros((nx, ny, full_shape_z), dtype = np.float32),
                        center = [cx, cy, full_center_z],
                        size = [sx, sy, full_size_z])
    grid_z_list = [get_grid_z(img) for img in img_list]

    def merge_img_(img_list, min_domi_radius = 45):
        center_z_list = [img.center[2] for img in img_list]
        
        min_center_z = np.min(center_z_list)
        max_center_z = np.max(center_z_list)
        full_size_z = max_center_z - min_center_z + sz
        full_center_z = (max_center_z + min_center_z) / 2
        full_shape_z = np.ceil(full_size_z / dz).astype(np.int)
        full_img = nef.Image(np.zeros((nx, ny, full_shape_z), dtype = np.float32), 
                            center = [cx, cy, full_center_z],
                            size = [sx, sy, full_size_z])
        grid_z_list = [get_grid_z(img) for img in img_list]
        full_grid_z = get_grid_z(full_img)
        for iz, z in enumerate(full_grid_z):
            i_img0, i_img1, dz0, dz1 = find_nearest_2_img(img_list, z)
            if i_img0 == -1:
                full_img.data[:,:,iz] = get_weight_slice(img_list[i_img1], z)
            elif i_img1 == -1:
                full_img.data[:,:,iz] = get_weight_slice(img_list[i_img0], z)
            else:
                dz0_ = dz0 - min_domi_radius
                dz1_ = dz1 - min_domi_radius
                if dz0 < min_domi_radius:
                    full_img.data[:,:,iz] = get_weight_slice(img_list[i_img0], z)
                elif dz1 < min_domi_radius:
                    full_img.data[:,:,iz] = get_weight_slice(img_list[i_img1], z)
                else:
                    w0, w1 = (dz1 - min_domi_radius) / (dz0 + dz1 - min_domi_radius * 2), (dz0 - min_domi_radius) / (dz0 + dz1 - min_domi_radius * 2)
                    full_img.data[:,:,iz] = get_weight_slice(img_list[i_img0], z) * w0 + get_weight_slice(img_list[i_img1], z) * w1
        return full_img
    
    return merge_img_(img_list)