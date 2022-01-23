import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from skimage.feature import greycoprops


def loadGeoTiff(root_image, init=None, size_img=None):
    tifsrc = gdal.Open(root_image)
    nbands = tifsrc.RasterCount
    in_band = tifsrc.GetRasterBand(1)
    if init is None:
        xinit,yinit = (0, 0)
    else:
        xinit,yinit = init
    if size_img is None:
        block_xsize, block_ysize = (in_band.XSize, in_band.YSize)
    else:
        block_xsize, block_ysize = size_img
    # read the multiband tile into a 3d numpy array
    image = tifsrc.ReadAsArray(xinit, yinit, block_xsize, block_ysize)
    image = np.moveaxis(image, 0, -1)
    return image, block_ysize, block_xsize, nbands


def RGBplot(image, index):
    img = np.dstack((image[:,:,index[0]]/np.percentile(image[:,:,index[0]],95),
                  image[:,:,index[1]]/np.percentile(image[:,:,index[1]],95),
                  image[:,:,index[2]]/np.percentile(image[:,:,index[2]],95)))
    img = np.clip(img, 0, 1)
    f = plt.figure()
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.show()

    
def scale(X, stast=None):
    cols = []
    descale = []
    X = X.T
    for feature in range(0, X.shape[0]):
        if stast is None:
            minimum = X[feature, :].min(axis=0)
            maximum = X[feature, :].max(axis=0)
        else:
            minimum = stast[feature][0]
            maximum = stast[feature][1]
        col_std = np.divide((X[feature, :] - minimum), (maximum - minimum))
        cols.append(col_std)
        descale.append((minimum, maximum))
    X_std = np.array(cols)
    return X_std.T, descale


def scale_std(X, stast=None):
    cols = []
    descale = []
    X = X.T
    for feature in range(0, X.shape[0]):
        if stast is None:
            mean = np.mean(X[feature, :],axis=0)
            standard = np.std(X[feature, :],axis=0)
        else:
            mean = stast[feature][0]
            standard = stast[feature][1]
        col_std = np.divide((X[feature, :] - mean), standard)
        cols.append(col_std)
        descale.append((mean, standard))
    X_std = np.array(cols)
    return X_std.T, descale


def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    # top
    val_ar.append(get_pixel(img, center, x - 1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def Local_binary_pattern(img_gray):
    height, width = img_gray.shape

    # Create a numpy array as
    # the same height and width
    # of RGB image
    img_lbp = np.zeros((height, width),
                       np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    return img_lbp


def offset(length, angle):
    """Return the offset in pixels for a given length and angle"""
    dv = length * np.sign(-np.sin(angle)).astype(np.int32)
    dh = length * np.sign(np.cos(angle)).astype(np.int32)
    return dv, dh


def crop(img, center, win):
    """Return a square crop of img centered at center (side = 2*win + 1)"""
    row, col = center
    side = 2*win + 1
    first_row = row - win
    first_col = col - win
    last_row = first_row + side
    last_col = first_col + side
    return img[first_row: last_row, first_col: last_col]


def cooc_maps(img, center, win, d=[1], theta=[0], levels=256):
    """
    Return a set of co-occurrence maps for different d and theta in a square
    crop centered at center (side = 2*w + 1)
    """
    shape = (2*win + 1, 2*win + 1, len(d), len(theta))
    cooc = np.zeros(shape=shape, dtype=np.int32)
    row, col = center
    Ii = crop(img, (row, col), win)
    for d_index, length in enumerate(d):
        for a_index, angle in enumerate(theta):
            dv, dh = offset(length, angle)
            Ij = crop(img, center=(row + dv, col + dh), win=win)
            cooc[:, :, d_index, a_index] = encode_cooccurrence(Ii, Ij, levels)
    return cooc


def encode_cooccurrence(x, y, levels=256):
    """Return the code corresponding to co-occurrence of intensities x and y"""
    return x*levels + y


def decode_cooccurrence(code, levels=256):
    """Return the intensities x, y corresponding to code"""
    return code//levels, np.mod(code, levels)


def compute_glcms(cooccurrence_maps, levels=256):
    """Compute the cooccurrence frequencies of the cooccurrence maps"""
    Nr, Na = cooccurrence_maps.shape[2:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na), dtype=np.float64)
    for r in range(Nr):
        for a in range(Na):
            codes, table = np.unique(cooccurrence_maps[:, :, r, a], return_counts=True)
            # codes = table[:, 0]
            freqs = table/float(table.sum())
            i, j = decode_cooccurrence(codes, levels=levels)
            glcms[i, j, r, a] = freqs
    return glcms


def compute_props(glcms, props=('contrast',)):
    """Return a feature vector corresponding to a set of GLCM"""
    Nr, Na = glcms.shape[2:]
    features = np.zeros(shape=(Nr, Na, len(props)))
    for index, prop_name in enumerate(props):
        features[:, :, index] = greycoprops(glcms, prop_name)
    return features.ravel()


def haralick_textures(img, win, d, theta, levels, props):
    """Return a map of Haralick features (one feature vector per pixel)"""
    rows, cols = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(d) * len(theta) * len(props)
    feature_map = np.zeros(shape=(rows, cols, n_features), dtype=np.float64)
    for m in range(rows):
        for n in range(cols):
            coocs = cooc_maps(arr, (m + margin, n + margin), win, d, theta, levels)
            glcms = compute_glcms(coocs, levels)
            feature_map[m, n, :] = compute_props(glcms, props)
    return feature_map
