import numpy as np
from scipy import sqrt, arctan2
from scipy.ndimage import uniform_filter
import scipy.io
from matplotlib import pyplot as plt
"""
   Extract Histogram of Oriented Gradients (HOG) for a given image 3D.
    Compute a Histogram of Oriented Gradients (HOG) by
        1. (optional) global image normalisation
        2. computing the gradient image in x and y
        3. computing gradient histograms
        4. normalising across blocks
        5. flattening into a feature vector
    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the HOG.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.
    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    hog_image : ndarray (if visualise=True)
        A visualisation of the HOG image.
    OBS:
    This code is a extend version of skimage HOG.
"""
def hog(image, orientations=36, pixels_per_cell=(9, 9, 9),
        cells_per_block=(9, 9, 9), visualise=False, normalise=False):
    if normalise:
        image = sqrt(image)
    if image.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers in np.diff()
        image = image.astype('float')

    gx = np.empty(image.shape, dtype=np.double)
    gx[:,0, 0] = 0
    gx[:,-1, -1] = 0
    gx[:,1:-1, 1:-1] = image[:,2:, 2:] - image[:,:-2, :-2]
    gy = np.empty(image.shape, dtype=np.double)
    gy[0, :, 0] = 0
    gy[-1, :, -1] = 0
    gy[1:-1, :, 1:-1] = image[2:, :, 2:] - image[:-2, :, :-2]

    gz = np.empty(image.shape, dtype=np.double)
    gz[0, 0, :] = 0
    gz[-1, -1, :] = 0
    gz[1:-1, 1:-1, :] = image[2:, 2:, :] - image[:-2, :-2, :]

    magnitude = sqrt(gx ** 2 + gy ** 2 + gz**2)
    orientation = arctan2(sqrt(gx**2 + gy**2),gz) * (180 / np.pi)
    sx, sy, sz = image.shape
    cx, cy, cz = pixels_per_cell
    bx, by, bz = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y
    n_cellsz = int(np.floor(sz // cz))

    orientation_histogram = np.zeros((n_cellsx, n_cellsy,n_cellsz, orientations))
    subsample = np.index_exp[cz // 2:cx * n_cellsz:cx,
                             cy // 2:cy * n_cellsy:cy,
                             cx // 2:cz * n_cellsx:cz]

    for i in range(orientations):
        temp_ori = np.where(orientation < 180.0 / orientations * (i + 1), orientation, -1)
        temp_ori = np.where(orientation >= 180.0 / orientations * i,
                            temp_ori, -1)
        cond2 = temp_ori > -1
        temp_mag = np.where(cond2, magnitude, 0)

        #temp_filt = uniform_filter(temp_mag, size=(cz, cy, cx))
        orientation_histogram[:, :, :, i] = temp_mag[subsample]


    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    n_blocksz = (n_cellsz - bz) + 1

    normalised_blocks = np.zeros((n_blocksz ,n_blocksy, n_blocksx, bz , by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            for z in range(n_blocksz):
                block = orientation_histogram[z:z+bz,y:y + by, x:x + bx, :]
            eps = 1e-5
            normalised_blocks[z,y, x, :] = block / sqrt(block.sum() ** 2 + eps)

    return normalised_blocks.ravel()
