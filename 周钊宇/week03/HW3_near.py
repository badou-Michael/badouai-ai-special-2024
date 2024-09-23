import cv2
import numpy as np
import matplotlib.pyplot as plt

def function(img):
    h,w,channels = img.shape
    empty_img = np.zeros((800,800,channels),np.uint8)
    hs = 800/h
    ws = 800/h
    for i in range(800):
        for j in range(800):
            x = int(i / hs + 0.5)
            y = int(j / ws + 0.5)
            empty_img[i, j] = img[x, y]
    return empty_img



img = cv2.imread("lenna.png")
# print(img)
zoom_img = function(img)
cv2.imshow("image", img)
cv2.imshow("zoom", zoom_img)
# img2 = cv2.resize(img, (800, 800), cv2.INTER_NEAREST)
# cv2.imshow("img2", img2)
cv2.waitKey(0)

#plt展示图片，像素值做了归一化，不适用上述方法
# img = plt.imread("lenna.png")
# # plt.subplot(221)
# # plt.imshow(img)
# print(img)
# print(img1 == img)
# zoom_img = function(img)
# # print(zoom_img)
# plt.subplot(222)
# plt.imshow(zoom_img)
# plt.show()