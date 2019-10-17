import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


def apply_sift(path_1, path_2):
    img = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    goodMatches = []
    while True:
        sift = cv2.xfeatures2d.SIFT_create()
        bf = cv2.BFMatcher()
        keyPoints1, descriptors1 = sift.detectAndCompute(gray, None)
        keyPoints2, descriptors2 = sift.detectAndCompute(gray_2, None)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                goodMatches.append(m)
        if len(goodMatches) <= 10:
            goodMatches = []
        else:
            break
    sourcePoints = np.float32([keyPoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 2)
    destinationPoints = np.float32([keyPoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 2)
    return gray, gray_2, sourcePoints, destinationPoints


def get_transform(p, q):
    A = np.array([[p[0, 0], p[0, 1], 1, 0, 0, 0, -p[0, 0] * q[0, 0], -p[0, 1] * q[0, 0]],
                  [0, 0, 0, p[0, 0], p[0, 1], 1, -p[0, 0] * q[0, 1], -p[0, 1] * q[0, 1]],
                  [p[1, 0], p[1, 1], 1, 0, 0, 0, -p[1, 0] * q[1, 0], -p[1, 1] * q[1, 0]],
                  [0, 0, 0, p[1, 0], p[1, 1], 1, -p[1, 0] * q[1, 1], -p[1, 1] * q[1, 1]],
                  [p[2, 0], p[2, 1], 1, 0, 0, 0, -p[2, 0] * q[2, 0], -p[2, 1] * q[2, 0]],
                  [0, 0, 0, p[2, 0], p[2, 1], 1, -p[2, 0] * q[2, 1], -p[2, 1] * q[2, 1]],
                  [p[3, 0], p[3, 1], 1, 0, 0, 0, -p[3, 0] * q[3, 0], -p[3, 1] * q[3, 0]],
                  [0, 0, 0, p[3, 0], p[3, 1], 1, -p[3, 0] * q[3, 1], -p[3, 1] * q[3, 1]]])
    b = np.array([q[0, 0], q[0, 1], q[1, 0], q[1, 1], q[2, 0], q[2, 1], q[3, 0], q[3, 1]])
    A_inv = np.linalg.inv(A)
    T = np.matmul(A_inv, b)
    return T


def apply_transform(pixel, T):
    x = (T[0]*pixel[0]+T[1]*pixel[1]+T[2]) / (T[6]*pixel[0]+T[7]*pixel[1]+1)
    y = (T[3]*pixel[0]+T[4]*pixel[1]+T[5]) / (T[6]*pixel[0]+T[7]*pixel[1]+1)
    res = np.array([x, y])
    return res


def apply_inverse_transform(pixel, T):
    y = ((pixel[1]-T[5])*(T[0]-T[6]*pixel[0])-(pixel[0]-T[2])*(T[3]-T[6]*pixel[1])) / \
        ((T[4]-T[7]*pixel[1])*(T[0]-T[6]*pixel[0])-(T[1]-T[7]*pixel[0])*(T[3]-T[6]*pixel[1]))
    x = (pixel[0]-y*(T[1]-T[7]*pixel[0])-T[2]) / (T[0]-T[6]*pixel[0])
    res = np.array([x, y])
    return res


def interpolate(pixel, img):
    x = pixel[0]
    y = pixel[1]
    p = int(np.floor(x)) % img.shape[0]
    p_plus = int(np.ceil(x)) % img.shape[0]
    q = int(np.floor(y)) % img.shape[1]
    q_plus = int(np.ceil(y)) % img.shape[1]
    a = x - p
    b = y - q
    return img[p, q]*(1-a)*(1-b) + img[p, q_plus]*(1-a)*b + img[p_plus, q]*a*(1-b) + img[p_plus, q_plus]*a*b


def ransac(src, dst, error, iterations):
    corrs = np.reshape(np.column_stack((src, dst)), (-1, 2, 2))
    best_T = np.zeros((3, 3))
    best_m = 0.0
    for i in range(iterations):
        transform_found = False
        T = np.zeros((3, 3))
        while not transform_found:
            try:
                np.random.shuffle(corrs)
                rand_src = corrs[:4, 0]
                rand_dst = corrs[:4, 1]
                T = get_transform(rand_src, rand_dst)
                transform_found = True
            except np.linalg.LinAlgError:
                continue
        transformed = np.apply_along_axis(apply_transform, 1, src, T)
        deltas = transformed - dst
        norms = np.linalg.norm(deltas, axis=1)
        inliers = norms[np.where(norms <= error)]
        m = (1/src.shape[0]) * len(inliers)
        if m > best_m:
            best_m = m
            best_T = T
        sys.stdout.write('\r')
        sys.stdout.write("Iteration: {} | Best Percentage: {}".format(str(i), best_m))
        sys.stdout.flush()
    sys.stdout.write('\n')
    return best_T


def fill_image(best_T, img, dst):
    bar = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-'
           , '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    total = dst.shape[0] * dst.shape[1]
    seen = 0
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            ind = (i, j)
            coords = apply_inverse_transform(ind, best_T)
            if 0 <= coords[0] < img.shape[0] and 0 <= coords[1] < img.shape[1]:
                dst[ind] = interpolate(coords, img)
            seen += 1
            percentage = int((seen/total) * 100)
            bar[percentage] = '#'
            bar_string = ''.join(bar)
            sys.stdout.write('\r')
            sys.stdout.write("Progress: {}{}{}".format('[',bar_string,']'))
            sys.stdout.flush()


# img, img_2, src, dst = apply_sift("casos\\caso_1\\1a.jpg", "casos\\caso_1\\1b.jpg")
# img, img_2, src, dst = apply_sift("casos\\caso_2\\2a.jpg", "casos\\caso_2\\2b.jpg")
img, img_2, src, dst = apply_sift("casos\\caso_3\\3a.jpg", "casos\\caso_3\\3b.jpg")
plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
plt.figure()
plt.imshow(img_2, cmap='gray')
plt.show()
best_T = ransac(src, dst, 5.0, 20000)
dst = np.append(np.zeros((img_2.shape[0], img.shape[1]), dtype=np.uint8), img_2, axis=1)
fill_image(best_T, img, dst)
plt.figure()
plt.imshow(dst, cmap='gray')
plt.show()


