from matplotlib import pyplot as plt
import numpy as np


def _astype_uint8(val: np.ndarray, ratio = 1):
    val = val - np.min(val)
    max_ = np.max(val)
    if max_ == 0:
        return np.zeros(val.shape, dtype = np.uint8)
    val1 = val / max_ * 255.999 * ratio
    filt1 = val < 0
    # filt2 = val > 255.999
    val1[filt1] = 0
    # val1[filt2] = 255.999
    return val1.astype(np.uint8)


def _gray2rgb(img: np.ndarray, **kwargs):
    rimg, gimg, bimg = [img * 0] * 3
    img = _astype_uint8(img, **kwargs)
    filt0, filt64, filt128, filt192 = img < 64, \
                                      (img >= 64) & (img < 128), \
                                      (img >= 128) & (img < 192), \
                                      img >= 192
    gimg[filt0] = img[filt0] * 4
    bimg[filt0] = 255

    gimg[filt64] = 255
    bimg[filt64] = 255 * 2 - 4 * img[filt64]

    rimg[filt128] = 4 * img[filt128] - 2 * 255
    gimg[filt128] = 255

    rimg[filt192] = 255
    gimg[filt192] = 255 * 4 - 4 * img[filt192]

    return np.dstack((rimg, gimg, bimg))


def fuse(img1: np.ndarray, img2: np.ndarray, ratio: float = 0.3, pattern: str = None):
    '''fuse img2 onto img1, with img1 as back gray image and img2 as rgb img'''
    if not img1.shape == img2.shape:
        raise ValueError('img1 and img2 should have same shape to do fusion')

    ratio1, ratio2 = ratio, 1 - ratio
    img1_rgb = _astype_uint8(np.dstack((img1, img1, img1)) * ratio1)
    img2_rgb = np.zeros(img2.shape + (3,), dtype = np.float32)
    if pattern is None:
        img2_rgb = _gray2rgb(img2, ratio = ratio2)

    return img2_rgb #img1_rgb# + img2_rgb