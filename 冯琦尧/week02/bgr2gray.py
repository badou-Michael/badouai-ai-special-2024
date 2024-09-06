import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def cv_bgr2gray(img):
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return dst

def split_gray(img, channel=1):
    b, g, r = cv2.split(img)
    if channel == 0:
        dst = b
    elif channel == 1:
        dst = g
    elif channel == 2:
        dst = r
    else:
        print("ERROR, channel must be 0, 1, 2")
    return dst

def mean_gray(img):
    dst = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3
    return dst.astype(np.uint8)

def max_gray(img):
    b, g, r = cv2.split(img)
    max_values = cv2.max(cv2.max(b, g), r)
    return max_values

def weight_avg_gray(img):
    dst = (img[:, :, 0] * 0.114 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.299).astype(np.uint8)
    return dst

if __name__ == "__main__":
    img = cv2.imread("lenna.png")  # origin image
    img_split_gray = split_gray(img, 0)  # bgr2gray by using split channel, 0->B, 1->G, 2->R

    img_mean_gray = mean_gray(img)  # average mean of BGR

    img_max_gray = max_gray(img)   # maximum value of BGR

    img_wa_gray = weight_avg_gray(img)  # weighted average value of BGR

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img.shape)

    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("IMG_BGR")

    plt.subplot(2, 3, 2)
    plt.imshow(img_rgb)
    plt.title("IMG_RGB")

    plt.subplot(2, 3, 3)
    plt.imshow(img_split_gray, cmap="gray")
    plt.title("SPLIT")

    plt.subplot(2, 3, 4)
    plt.imshow(img_mean_gray, cmap="gray")
    plt.title("MEAN")

    plt.subplot(2, 3, 5)
    plt.imshow(img_max_gray, cmap="gray")
    plt.title("MAX")

    plt.subplot(2, 3, 6)
    plt.imshow(img_wa_gray, cmap="gray")
    plt.title("WEIGHTED_AVG")

    plt.show()
