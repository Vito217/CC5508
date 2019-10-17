import numpy as np
import skimage.io as skio
import scipy.ndimage.filters as nd_filters
import matplotlib.pyplot as plt
import skimage.feature as feature
import os
import argparse


def get_gradient_per_pixel(filename, canny=False, sigma=3):
    """
    Returns image, gradients, angles and magnitudes per pixel as numpy arrays
    :param filename: path to the
    :param canny: True if uses canny
    :param sigma: Sigma used in canny
    :return: image_array, gradient_x_array, gradient_y_array, angles_array, magnitudes_array
    """

    image = skio.imread(filename, as_gray=True)
    if canny:
        image = feature.canny(image, sigma)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)

    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    ang, mag = np.arctan2(gy, gx), np.sqrt(np.square(gx) + np.square(gy))
    ang[ang < 0] += np.pi

    return image, gx, gy, ang, mag


def interpolation(b_mag, b_ang, k):
    """
    Returns histogram of orientations per block, using linear interpolation
    :param b_mag: Magnitudes per block, as numpy array
    :param b_ang: Angles per block, as numpy array
    :param k: Number of bins
    :return: histogram_array
    """

    # Filling histogram
    ind = b_ang * k / np.pi
    left = np.floor(ind - 0.5)
    right = left + 1
    wl = ind - left - 0.5
    wr = 1 - wl
    left = left.astype(np.int) % k
    right = right.astype(np.int) % k

    h = np.zeros(k, np.float32)
    for i in range(k):
        l_rows, l_cols = np.where(left == i)
        r_rows, r_cols = np.where(right == i)
        h[i] = np.sum(b_mag[l_rows, l_cols] * wl[l_rows, l_cols]) + np.sum(b_mag[r_rows, r_cols] * wr[r_rows, r_cols])
    h = h / np.linalg.norm(h, 2)

    return h


def hog(filename, k, canny=False, sigma=3):
    """
    Returns histogram of gradients per pixel, with K bins
    :param filename: Path to image
    :param k: Number of bins
    :param canny: True if uses canny
    :param sigma: Sigma used in canny
    :return: histogram_array
    """

    # Getting gradient per pixel
    image, gx, gy, ang, mag = get_gradient_per_pixel(filename, canny, sigma)

    # Getting indexes
    ind = np.round(ang * k / np.pi).astype(np.int) % k

    # Filling histogram
    h = np.zeros(k, np.float32)
    for i in range(k):
        rows, cols = np.where(ind == i)
        h[i] = np.sum(mag[rows, cols])
    h = h / np.linalg.norm(h, 2)

    return h, image, ang, mag


def helo(filename, k, b, canny=False, sigma=3):
    """
    Returns histogram of gradients for each block, with K bins
    :param filename: Path to image
    :param k: Number of bins
    :param b: Number of blocks (rows and columns)
    :param canny: True if uses canny
    :param sigma: Sigma used in canny
    :return: histogram_array
    """

    # Getting gradient per pixel
    image, gx, gy, ang, mag = get_gradient_per_pixel(filename, canny, sigma)

    # Getting gradient per block
    b_gx = np.square(gx) - np.square(gy)
    b_gy = 2 * gx * gy

    # Initializing Lx, Ly and magnitudes per block
    lx = np.zeros((b, b), np.float32)
    ly = np.zeros((b, b), np.float32)
    b_mag = np.zeros((b, b), np.float32)

    # Rows and cols where every pixel is
    i = np.transpose(np.tile(np.arange(image.shape[0]), (image.shape[1], 1)))
    j = np.tile(np.arange(image.shape[1]), (image.shape[0], 1))

    # Block indexes for each pixel
    r_ind = np.round(i * b / image.shape[0]).astype(np.int) % b
    s_ind = np.round(j * b / image.shape[1]).astype(np.int) % b

    # Filling Lx, Ly and Magnitudes per block
    for r in range(b):
        for s in range(b):
            rows, cols = np.where((r_ind == r) & (s_ind == s))
            lx[r, s] = np.sum(b_gx[rows, cols])
            ly[r, s] = np.sum(b_gy[rows, cols])
            b_mag[r, s] = np.sum(mag[rows, cols])

    # Getting angles
    b_ang = (np.arctan2(ly, lx) + np.pi) * 0.5

    return interpolation(b_mag, b_ang, k), image, b_ang, b_mag


def shelo(filename, k, b, canny=False, sigma=3):
    """
    Soft version of HELO
    :param filename: Path to image
    :param k: Number of bins
    :param b: Number of blocks (rows and columns)
    :param canny: True if uses canny
    :param sigma: Sigma used in canny
    :return: histogram_array
    """

    # Getting gradient per pixel
    image, gx, gy, ang, mag = get_gradient_per_pixel(filename, canny, sigma)

    # Getting gradient per block
    b_gx = np.square(gx) - np.square(gy)
    b_gy = 2 * gx * gy

    # Initializing Lx, Ly and Magnitudes per block
    lx = np.zeros((b, b), np.float32)
    ly = np.zeros((b, b), np.float32)
    b_mag = np.zeros((b, b), np.float32)

    # Rows and cols where every pixel is
    i = np.transpose(np.tile(np.arange(image.shape[0]), (image.shape[1], 1)))
    j = np.tile(np.arange(image.shape[1]), (image.shape[0], 1))

    # Block indexes for each pixel
    r_ind = i * b / image.shape[0]
    s_ind = j * b / image.shape[1]

    # Left, Right, Top and Bottom indexes
    left = np.floor(s_ind - 0.5)
    right = left + 1
    top = np.floor(r_ind - 0.5)
    bottom = top + 1

    # Left, Right, Top and Bottom weights
    d_left = s_ind - left - 0.5
    d_right = 1 - d_left
    d_top = r_ind - top - 0.5
    d_bottom = 1 - d_top
    w_left = 1 - d_left
    w_right = 1 - d_right
    w_top = 1 - d_top
    w_bottom = 1 - d_bottom

    left = np.round(left).astype(np.int) % b
    right = np.round(right).astype(np.int) % b
    top = np.round(top).astype(np.int) % b
    bottom = np.round(bottom).astype(np.int) % b

    r_ind = np.round(r_ind).astype(np.int) % b
    s_ind = np.round(s_ind).astype(np.int) % b

    for r in range(b):
        for s in range(b):

            # Getting lx and ly
            lt_rows, lt_cols = np.where((left == s) & (top == r))
            rt_rows, rt_cols = np.where((right == s) & (top == r))
            lb_rows, lb_cols = np.where((left == s) & (bottom == r))
            rb_rows, rb_cols = np.where((right == s) & (bottom == r))

            lx[r, s] = np.sum(b_gx[lt_rows, lt_cols] * w_left[lt_rows, lt_cols] * w_top[lt_rows, lt_cols]) + \
                       np.sum(b_gx[rt_rows, rt_cols] * w_right[rt_rows, rt_cols] * w_top[rt_rows, rt_cols]) + \
                       np.sum(b_gx[lb_rows, lb_cols] * w_left[lb_rows, lb_cols] * w_bottom[lb_rows, lb_cols]) + \
                       np.sum(b_gx[rb_rows, rb_cols] * w_right[rb_rows, rb_cols] * w_bottom[rb_rows, rb_cols])

            ly[r, s] = np.sum(b_gy[lt_rows, lt_cols] * w_left[lt_rows, lt_cols] * w_top[lt_rows, lt_cols]) + \
                       np.sum(b_gy[rt_rows, rt_cols] * w_right[rt_rows, rt_cols] * w_top[rt_rows, rt_cols]) + \
                       np.sum(b_gy[lb_rows, lb_cols] * w_left[lb_rows, lb_cols] * w_bottom[lb_rows, lb_cols]) + \
                       np.sum(b_gy[rb_rows, rb_cols] * w_right[rb_rows, rb_cols] * w_bottom[rb_rows, rb_cols])

            # Getting magnitudes
            rows, cols = np.where((r_ind == r) & (s_ind == s))
            b_mag[r, s] = np.sum(mag[rows, cols])

    # Getting angles
    b_ang = (np.arctan2(ly, lx) + np.pi) * 0.5

    return interpolation(b_mag, b_ang, k), image, b_ang, b_mag


def query(path, k, b, hist_type, label=None, canny=False, sigma=3):
    """
    Returns histogram and label from query image
    :param path: Path to image
    :param k: Number of bins
    :param b: Number of cells per row
    :param hist_type: hog, helo or shelo
    :param label: label of image
    :return: pair list of histogram and label
    """

    if hist_type == "hog":
        hist, image, ang, mag = hog(path, k, canny, sigma)
    elif hist_type == "helo":
        hist, image, ang, mag = helo(path, k, b, canny, sigma)
    else:
        hist, image, ang, mag = shelo(path, k, b, canny, sigma)

    path_words = path.split("/")
    label = label if label is not None else path_words[len(path_words)-1]

    return [hist, label]


def create_data_set(data_path, k, b, hist_type, canny=False, sigma=3):
    """
    Creates numpy histograms from dataset
    :param data_path: path to dataset folder
    :param k: Number of bins
    :param b: Number of cells per row
    :param hist_type: hog, helo or shelo
    :param canny: True if uses canny
    :param sigma: Sigma used in canny
    :return: Numpy histogram matrix
    """

    print("Loading data from {}".format(data_path))

    data_plus_labels = []

    for dirpath, dirnames, filenames in os.walk(data_path):
        for file in [f for f in filenames if f.endswith(".jpg")]:

            path = os.path.join(dirpath, file)
            label = dirpath.split("\\")[-1]

            print("Reading {} with label '{}'".format(path, label))

            data_plus_labels.append(query(path, k, b, hist_type, label, canny, sigma))

    # Saving data
    data = np.array(data_plus_labels)
    if hist_type == "hog":
        save_path = "tarea_2/npy_data/data" + ("_canny_{}_k_{}" if canny else "_{}_k_{}")
        np.save(save_path.format(hist_type, k), data)
        print("File successfully saved as queries_{}_k_{}.npy".format(hist_type, k), data)
    else:
        save_path = "tarea_2/npy_data/data" + ("_canny_{}_k_{}_b_{}" if canny else "_{}_k_{}_b_{}")
        np.save(save_path.format(hist_type, k, b), data)
        print("File successfully saved as queries_{}_k_{}_b_{}.npy".format(hist_type, k, b), data)

    return data


def create_query_set(map_path, queries_path, k, b, hist_type, canny=False, sigma=3):
    """
    Creates numpy histograms from query set
    :param map_path: path to label mapping
    :param queries_path: path to queries folder
    :param k: Number of bins
    :param b: Number of cells per row
    :param hist_type: hog, helo or shelo
    :param canny: True if uses canny
    :param sigma: Sigma used in canny
    :return: Numpy histogram matrix
    """

    print("Loading queries from {} \n".format(queries_path))

    data_plus_labels = []

    with open(map_path, "r") as f:
        for line in f:

            name, label = line.split("\t")
            label = label.replace("\n", "")
            path = (queries_path+"\\"+name).replace("\n", "")

            print("Reading {} with label '{}'".format(path, label))

            data_plus_labels.append(query(path, k, b, hist_type, label, canny, sigma))

    # Saving data
    data = np.array(data_plus_labels)
    if hist_type == "hog":
        save_path = "tarea_2/npy_data/queries" + ("_canny_{}_k_{}" if canny else "_{}_k_{}")
        np.save(save_path.format(hist_type, k), data)
        print("File successfully saved as queries_{}_k_{}.npy".format(hist_type, k), data)
    else:
        save_path = "tarea_2/npy_data/queries" + ("_canny_{}_k_{}_b_{}" if canny else "_{}_k_{}_b_{}")
        np.save(save_path.format(hist_type, k, b), data)
        print("File successfully saved as queries_{}_k_{}_b_{}.npy".format(hist_type, k, b), data)

    return data


def load_numpy(path):
    """
    Reads numpy file
    :param path: path to file
    :return: numpy array
    """

    return np.load(path)


def euclidean_distance(x_1, x_2):
    """
    Computes the euclidean distance
    :param x_1: numpy array
    :param x_2: numpy array
    :return: distance as a float number
    """

    return np.sqrt(np.sum(np.square(x_1 - x_2)))


def distance_vector(q, dataset, n):
    """
    Returns a pair vector. First element is expected label. Second is list of N closest.
    :param q: query vector
    :param dataset: numpy matrix of data
    :param n: number of results
    :return: pair list
    """

    vec = []
    query_hist, expected_label = q[0], q[1]
    for data_hist, real_label in dataset:
        vec.append([euclidean_distance(query_hist, data_hist), real_label])
    vec = np.array(vec)
    vec = vec[vec[:, 0].argsort()]

    return [expected_label.replace("\n", ""), vec[0:n, 1]]


def distance_matrix(queryset, dataset, n):
    """
    Returns matrix of distance vectors
    :param queryset: Numpy Matrix
    :param dataset: Numpy Matrix
    :param n: Number of results
    :return: list
    """

    m = []
    for q in queryset:
        dv = distance_vector(q, dataset, n)
        print(dv)
        m.append(dv)
    return m


def mean_average_precision(dist_matrix):
    """
    Computes the mAP out of a matrix
    :param dist_matrix: numpy matrix
    :return: float number
    """

    total_queries = len(dist_matrix)
    mean_avg_precision = 0.0

    for expected, results in dist_matrix:
        total_seen = 0.0
        total_relevant = 0.0
        precision = 0.0
        for result in results:
            total_seen += 1.0
            if result == expected:
                total_relevant += 1.0
                precision += total_relevant / total_seen
        avg_precision = (precision / total_relevant) if total_relevant > 0 else 0
        mean_avg_precision += avg_precision / total_queries

    return mean_avg_precision


def plot_gradients(filename, k, b, hist_type):
    """
    Plots the image, gradients and histogram
    :param filename: path to image
    :param k: Number of bins
    :param b: Number of cells per row
    :param hist_type: hog, helo or shelo
    :return: none
    """

    print("Please Wait...")

    if hist_type == "hog":
        hist, image, ang, mag = hog(filename, k)
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].imshow(image, cmap='gray', extent=[0, image.shape[1], image.shape[0], 0])
        ax[1].bar(range(k), hist)

    else:
        if hist_type == "helo":
            hist, image, ang, mag = helo(filename, k, b)
        else:
            hist, image, ang, mag = shelo(filename, k, b)

        fig, ax = plt.subplots(1, 3, figsize=(14, 6))
        ax[0].imshow(image, cmap='gray', extent=[0, image.shape[1], image.shape[0], 0])
        ax[1].imshow(image, cmap='gray', extent=[0, image.shape[1], image.shape[0], 0])
        ax[2].bar(range(k), hist)

        block_width = image.shape[1] / b
        block_height = image.shape[0] / b

        for r in range(b):
            for s in range(b):

                b_ang = ang[r, s]
                m = np.tan(b_ang)
                if m == 0:
                    if b_ang == 0:
                        h = 0
                        w = block_width/2
                    else:
                        h = 0
                        w = -block_width/2
                else:
                    h = (block_width / 2) * m
                    if np.abs(h) > block_height / 2:
                        h = block_height / 2
                    w = h / m
                    h = np.abs(h)

                x = np.array([(s*block_width+(block_width/2-np.abs(w))), (s*block_width+(block_width/2+np.abs(w)))])
                if w > 0:
                    y = np.array([(r*block_height+block_height/2-h), (r*block_height+block_height/2+h)])
                else:
                    y = np.array([(r*block_height+block_height/2+h), (r*block_height+block_height/2-h)])

                ax[1].plot(x, y, '-', linewidth=1.5, color='yellow')

    print("Plot ready")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encoding text on images")
    parser.add_argument("--canny", action='store_true')
    parser.add_argument("--mAP", action='store_true')
    parser.add_argument("--histogram", type=str, help="hog, helo or shelo", required=False)
    parser.add_argument("--image", type=str, help="Path to image", required=False)
    parser.add_argument("--k", type=int, help="Number of bins", required=False)
    parser.add_argument("--b", type=int, help="Number of blocks", required=False)
    parser.add_argument("--n", type=int, help="Number of query results", required=False)
    parser.add_argument("--query", action='store_true')
    parser.add_argument("--label", type=str, help="Query label", required=False)
    parser.add_argument("--sigma", type=int, help="Canny sigma", required=False)
    parser.add_argument("--dload", type=str, help="Path to data", required=False)
    parser.add_argument("--qload", type=str, help="Path to queries", required=False)
    parser.add_argument("--qmap", type=str, help="Path to queries mapping", required=False)

    pargs = parser.parse_args()

    if pargs.histogram and pargs.image and ((pargs.k and pargs.b) or (pargs.k and pargs.histogram == "hog")):

        if pargs.query:
            if pargs.histogram=="hog":
                npy_path = "tarea_2/npy_data/data" + ("_canny_{}_k_{}.npy" if pargs.canny else "_{}_k_{}.npy")
            else:
                npy_path = "tarea_2/npy_data/data"+("_canny_{}_k_{}_b_{}.npy" if pargs.canny else "_{}_k_{}_b_{}.npy")
            label = pargs.label if pargs.label else None
            n = pargs.n if pargs.n else 20
            canny = True if pargs.canny else False
            sigma = pargs.sigma if pargs.sigma else 3

            d = load_numpy(npy_path.format(pargs.histogram, pargs.k, pargs.b))
            q = query(pargs.image, pargs.k, pargs.b, pargs.histogram, label, canny, sigma)
            print(distance_vector(q, d, n))
        else:
            plot_gradients(pargs.image, pargs.k, pargs.b, pargs.histogram)

    elif pargs.histogram and (pargs.dload or pargs.qload) and \
            ((pargs.k and pargs.b) or (pargs.k and pargs.histogram == "hog")):

        canny = True if pargs.canny else False
        sigma = pargs.sigma if pargs.sigma else 3

        if pargs.dload:
            create_data_set(pargs.dload, pargs.k, pargs.b, pargs.histogram, canny, sigma)
        elif pargs.qload and pargs.qmap:
            create_query_set(pargs.qmap, pargs.qload, pargs.k, pargs.b, pargs.histogram, canny, sigma)
        else:
            print("Error: missing some arguments")

    elif pargs.mAP and pargs.n and pargs.histogram and \
            ((pargs.k and pargs.b) or (pargs.k and pargs.histogram == "hog")):

        if pargs.histogram=="hog":
            file = "_{}_k_{}.npy".format(pargs.histogram, pargs.k)
        else:
            file = "_{}_k_{}_b_{}.npy".format(pargs.histogram, pargs.k, pargs.b)

        if pargs.canny:
            file = "_canny"+file

        d = load_numpy("tarea_2/npy_data/data"+file)
        q = load_numpy("tarea_2/npy_data/queries"+file)
        print("mAP = {}".format(mean_average_precision(distance_matrix(q, d, pargs.n))))
    else:
        print("Error: missing some arguments")