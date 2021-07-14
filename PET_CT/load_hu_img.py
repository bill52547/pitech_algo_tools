import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import pydicom
import os
import srfnef as nef
from scipy.ndimage import zoom

foldername = '/mnt/nfs/decoder/exp_08_31/CT/01797532/'

import os
files = []
for i, j, k in os.walk(foldername):
    files = list(k)

off_z = np.zeros(len(files), dtype = np.float32)
for ind, file in enumerate(files):
    dataset = pydicom.dcmread(foldername + file)
    off_z[ind] = float(dataset.SliceLocation)
sort_z = np.argsort(off_z)


center = [0, 0, (off_z[0] + off_z[-1]) / 2]
dx, dy = [float(x) for x in dataset.PixelSpacing]
dz = float(dataset.SliceThickness)
nx = int(dataset.Rows)
ny = int(dataset.Columns)
nz = int(len(files))
size = [nx * dx, ny * dy, nz * dz]
hu_img_data = np.zeros((nx, ny, nz), dtype = np.int16)
for iz in range(len(files)):
    dataset = pydicom.dcmread(foldername + files[sort_z[iz]])
    hu_img_data[:,:,iz] = dataset.pixel_array