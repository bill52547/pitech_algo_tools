import numpy as np
import srfnef as nef
from matplotlib import pyplot as plt
import json
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    '''calculate rigid matrix and transportation of two set of points'''
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


import os
def get_pnt_pos(foldername):
    '''get big and small point locations from PET images'''
    img_path = os.path.join(foldername, 'result_data.bin.hdf5')
    json_path = os.path.join(foldername, 'result_meta.json')
    
    pet_img = nef.load(nef.Image, img_path)
    with open(json_path, 'r') as fin:
        dct = json.load(fin)
    m14, m24, m34 = dct['affine']['m14'], dct['affine']['m24'], dct['affine']['m34']
    
    ix, iy, iz = [x_[0] for x_ in np.where(pet_img.data > np.max(pet_img.data) * 0.99)]
    xx, yy, zz = pet_img.voxel_pos()
    big_pos = np.array([xx[ix, iy, iz] + m24, yy[ix, iy, iz] + m14, zz[ix, iy, iz] + m34])
    
    pet_img.data[ix - 5: ix + 5, iy - 5: iy + 5, iz - 5: iz + 5] = 0
    ix, iy, iz = [x_[0] for x_ in np.where(pet_img.data > np.max(pet_img.data) * 0.99)]
    xx, yy, zz = pet_img.voxel_pos()
    sml_pos = np.array([xx[ix, iy, iz] + m24, yy[ix, iy, iz] + m14, zz[ix, iy, iz] + m34])

    return big_pos, sml_pos


def pick_inds(tag_list, pick_order =['hx', 'g_', 'ny', 'cgy', 'vince']):
    '''calculate pick order index'''
    for pick_str in pick_order:
        for ind, tag in enumerate(tag_list):
            if tag.startswith(pick_str):
                return ind

def get_by_weight(df, weight: str = '', offsets = [0, -1100, -1400]):
    '''get PET and CT pnt locations with center as offsets'''
    ct_pos, pet_pos = np.zeros((0, 3), dtype = np.float32), np.zeros((0, 3), dtype = np.float32)
    weight_filt = df['description'].str.startswith(weight)
    offx, offy, offz = offsets
    for prot_id in np.unique(df['protocol_id']):
        prot_filt = df['protocol_id'] == prot_id
        for type_ in ['large', 'small']:
            type_filt = df['ct_point_type'] == type_
            ct_tag_rep_filt = weight_filt & prot_filt & type_filt
            if np.sum(ct_tag_rep_filt) > 1:
                ct_tag, ct_x, ct_y, ct_z, pet_x, pet_y, pet_z = [np.array(df[key][ct_tag_rep_filt]) for key in ['ct_tag', 'ct_x', 'ct_y', 'ct_z', 'pet_x', 'pet_y', 'pet_z']]
                ind = pick_inds(ct_tag)
                ct_pos = np.vstack((ct_pos, [ct_x[ind] + offx, ct_y[ind] + offy, ct_z[ind] + offz]))
                pet_pos = np.vstack((pet_pos, [pet_x[ind] + offx, pet_y[ind] + offy, pet_z[ind] + offz]))
   
            elif np.sum(ct_tag_rep_filt) == 1:
                ct_tag, ct_x, ct_y, ct_z, pet_x, pet_y, pet_z = [np.array(df[key][ct_tag_rep_filt]) for key in ['ct_tag', 'ct_x', 'ct_y', 'ct_z', 'pet_x', 'pet_y', 'pet_z']]
                ct_pos = np.vstack((ct_pos, [ct_x[ind] + offx, ct_y[ind] + offy, ct_z[ind] + offz]))
                pet_pos = np.vstack((pet_pos, [pet_x[ind] + offx, pet_y[ind] + offy, pet_z[ind] + offz]))
    return ct_pos, pet_pos

df = pd.read_csv('/media/nvme0/programs/guo/exp_2021_07_07_pnt_source_reg/Result_219.csv')

R_arr = np.zeros((5, 3, 3))
C_arr = np.zeros((5, 3))
for ind, weight in enumerate(['0', '10kg', '50kg', '70kg', '90kg']):
    ct_pos, pet_pos = get_by_weight(df, weight)
    R, C = rigid_transform_3D(ct_pos.T, pet_pos.T)
    C_arr[ind, :] = C.ravel()
    R_arr[ind, :, :] = R


'''plot error curve'''
fig, ax = plt.subplots(figsize = (15, 5))
plt.plot(C_arr[:, 0], 'b*-', label = 'x')
plt.plot([np.mean(C_arr[:, 0])] * 5, 'b-.', label = 'x-mean')
plt.plot(C_arr[:, 1], 'g*-', label = 'y')
plt.plot([np.mean(C_arr[:, 1])] * 5, 'g-.', label = 'y-mean')
plt.plot(C_arr[:, 2], 'r*-', label = 'z')
plt.plot([np.mean(C_arr[:, 2])] * 5, 'r-.', label = 'z-mean')
x_diff = C_arr[:, 0] - np.mean(C_arr[:, 0])
y_diff = C_arr[:, 1] - np.mean(C_arr[:, 1])
z_diff = C_arr[:, 2] - np.mean(C_arr[:, 2])
print(f'x diff {np.round(x_diff, decimals=2)} mm', f', std = {np.std(x_diff):.2f} mm, max err = {np.max(x_diff):.2f} mm')
print(f'y diff {np.round(y_diff, decimals=2)} mm', f', std = {np.std(y_diff):.2f} mm, max err = {np.max(y_diff):.2f} mm')
print(f'z diff {np.round(z_diff, decimals=2)} mm', f', std = {np.std(z_diff):.2f} mm, max err = {np.max(z_diff):.2f} mm')

print(f'mean offsets = {np.mean(C_arr[:, 0]):.2f} {np.mean(C_arr[:, 1]):.2f} {np.mean(C_arr[:, 2]):.2f} mm')
ax.set_xticks([0, 1,2,3,4])
ax.set_xticklabels(['0kg', '10kg', '50kg', '70kg', '90kg'])
plt.legend()

'''print error according to sets from weights'''

weight_list = ['', '0', '10kg', '50kg', '70kg', '90kg']
for ind, weight in enumerate(weight_list):
    ct_pos, pet_pos = get_by_weight(df, weight)
    for ind2 in range(5):
        R, C = R_arr[ind2, :, :], C_arr[ind2, :]
        err = pet_pos - ct_pos - C
        err_3d_mean = np.mean(np.sum(err ** 2, axis = 1) ** 0.5)
        err_3d_max = np.max(np.sum(err ** 2, axis = 1) ** 0.5)
        
        err_x_mean = np.mean(np.abs(err[:, 0]))
        err_x_max = np.max(np.abs(err[:, 0]))
        err_y_mean = np.mean(np.abs(err[:, 1]))
        err_y_max = np.max(np.abs(err[:, 1]))
        err_z_mean = np.mean(np.abs(err[:, 2]))
        err_z_max = np.max(np.abs(err[:, 2]))
        
        err = pet_pos - (R @ ct_pos.T).T - C
        err_3d_mean_r = np.mean(np.sum(err ** 2, axis = 1) ** 0.5)
        err_3d_max_r = np.max(np.sum(err ** 2, axis = 1) ** 0.5)
        
        err_x_mean_r = np.mean(np.abs(err[:, 0]))
        err_x_max_r = np.max(np.abs(err[:, 0]))
        err_y_mean_r = np.mean(np.abs(err[:, 1]))
        err_y_max_r = np.max(np.abs(err[:, 1]))
        err_z_mean_r = np.mean(np.abs(err[:, 2]))
        err_z_max_r = np.max(np.abs(err[:, 2]))
        
        
#         print('using C from ', weight, f'to cal {weight_list[ind2]}, the mean 3D err {err_3d_mean:.3f} mm, the max 3D err {err_3d_max:.3f} mm')
#         print('using R, C from ', weight, f'to cal {weight_list[ind2]}, the mean 3D err {err_3d_mean_r:.3f} mm, the max 3D err {err_3d_max_r:.3f} mm')
        
        print('using C from ', weight, f'to {weight_list[ind2]}, mean: x {err_x_mean:.3f} y {err_y_mean:.3f} z {err_z_mean:.3f}, max: x {err_x_max:.3f} y {err_y_max:.3f} z {err_z_max:.3f}')
        print('using R, C from ', weight, f'to {weight_list[ind2]}, mean: x {err_x_mean_r:.3f} y {err_y_mean_r:.3f} z {err_z_mean_r:.3f}, max: x {err_x_max_r:.3f} y {err_y_max_r:.3f} z {err_z_max_r:.3f}')
