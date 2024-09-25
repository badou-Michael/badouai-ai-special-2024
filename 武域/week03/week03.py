import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image
import cv2

lenna = cv2.imread('lenna.png')
h, w, l= lenna.shape
lenna_gray = rgb2gray(lenna)
# 临近差值
def nearest_neighbor_interpolation (image, h0, w0):
    h, w, l= image.shape
    print(f"Converting from {h} by {w} to {h0} by {w0}.")
    if h == h0 and w==w0:
        print("image size not changed")
        return image
    
    new_image = np.zeros((h0, w0, l), dtype=np.uint8)
    row_ratio = float(h) / h0
    col_ratio = float(w) / w0
    print(row_ratio, col_ratio)

    for i in range(h0):
        for j in range(w0):
            x = round(i*row_ratio)
            y = round(j*col_ratio)
            
            x = min(x, h - 1)
            y = min(y, w - 1)
            new_image[i, j] = image[x, y]

    return new_image

# test
# resized_lenna = nearest_neighbor_interpolation(lenna, 1080, 1920)

# cv2.imshow('lenna',resized_lenna)
# cv2.waitKey(0)

# 双线性插值（bilinear interpolation)
def bilinear_interpolation(image, destH, destW):
    origH, origW, layer = image.shape
    new_image = np.zeros((destH, destW, layer), dtype=np.uint8)
    scaleX = float(origW) / destW
    scaleY = float(origH) / destH

    for i in range(layer):
        for destY in range(destH):
            for destX in range(destW):
                # 计算原图像中的坐标，中心对齐
                origX = (destX + 0.5) * scaleX - 0.5
                origY = (destY + 0.5) * scaleY - 0.5

                # 找到相邻的4个像素的坐标
                x0 = int(np.floor(origX))
                x1 = min(x0 + 1, origW - 1)
                y0 = int(np.floor(origY))
                y1 = min(y0 + 1, origH - 1)

                # 双线性插值
                temp0 = (x1 - origX) * image[y0, x0, i] + (origX - x0) * image[y0, x1, i]
                temp1 = (x1 - origX) * image[y1, x0, i] + (origX - x0) * image[y1, x1, i]
                new_image[destY, destX, i] = int((y1 - origY) * temp0 + (origY - y0) * temp1)


    print(f"Total pixels processed: {pix}")
    return new_image

# resized_lenna = bilinear_interpolation(lenna, 1080, 1920)

# cv2.imshow('lenna',resized_lenna)
# cv2.waitKey(0)

# histogram equalization
def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    cdf_normalized = 255 * (cdf - cdf.min()) / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    equalized_image = cdf_normalized[image.flatten()].reshape(image.shape)

    return equalized_image

lenna_pro = histogram_equalization(lenna)

cv2.imshow('lenna',lenna_pro)
cv2.waitKey(0)
