import numpy as np
from scipy.interpolate import interpn, RegularGridInterpolator

def my_zoom2d(img, out_shape: tuple):
    nx, ny = img.shape
    if isinstance(out_shape, int):
        ratio = out_shape
        out_shape = (nx * ratio, ny * ratio)
    nx2, ny2 = out_shape
    x1, y1 = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    x2, y2 = np.linspace(0, 1, nx2), np.linspace(0, 1, ny2)
    xx2, yy2 = np.meshgrid(x2, y2, indexing='ij')
    xx2, yy2 = xx2.ravel(), yy2.ravel()
    interp = RegularGridInterpolator((x1, y1), img, method = 'linear', bounds_error = False, fill_value = 0)
    return interp(np.vstack((xx2, yy2)).T).reshape(out_shape)
    
def my_zoom3d(img, out_shape: tuple):
    nx, ny, nz = img.shape
    if isinstance(out_shape, int):
        ratio = out_shape
        out_shape = (nx * ratio, ny * ratio, nz * ratio)
    nx2, ny2, nz2 = out_shape
    x1, y1, z1 = np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz)
    x2, y2, z2 = np.linspace(0, 1, nx2), np.linspace(0, 1, ny2), np.linspace(0, 1, nz2)
    xx2, yy2, zz2 = np.meshgrid(x2, y2, z2, indexing='ij')
    xx2, yy2, zz2 = xx2.ravel(), yy2.ravel(), zz2.ravel()
    interp = RegularGridInterpolator((x1, y1, z1), img, method = 'linear', bounds_error = False, fill_value = 0)
    return interp(np.vstack((xx2, yy2, zz2)).T).reshape(out_shape)
    