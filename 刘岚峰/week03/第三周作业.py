import cv2
import numpy as np



# 最邻近插值
def ne_interpolate(img, sh, sw):
    h, w, c = img.shape
    empty_img = np.zeros((sh, sw, c), np.uint8)
    ax = h / sh
    ay = w / sw
    for i in range(sh):
        for j in range(sw):
            x = int(i * ax + 0.5)
            y = int(j * ay + 0.5)
            empty_img[i, j] = img[x, y]
    return empty_img



# 双线性插值
def bl_interpolate(img, sh, sw):
    h, w, c = img.shape
    empty_img = np.zeros((sh, sw, c), np.uint8)
    ax = h / sh
    ay = w / sw
    for i in range(sh):
        for j in range(sw):
            x = (i + 0.5) * ax - 0.5
            y = (j + 0.5) * ay - 0.5
            ix = int(np.floor(x))
            iy = int(np.floor(y))
            dx = x - ix
            dy = y - iy
            temp0 = (1 - dx) * img[ix, iy] + dx * img[min(ix + 1, h - 1), iy]
            temp1 = (1 - dx) * img[ix, min(iy + 1, w - 1)] + dx * img[min(ix + 1, h - 1), min(iy + 1, w - 1)]
            empty_img[i, j] = (1-dy) * temp0 + dy * temp1
    return empty_img
"""
几何中心重合+0.5证明
原图像M*M,目标图像N*N
原图几何中心【(M-1)/2,(M-1)/2】
目标图像几何中心【(N-1)/2,(N-1)/2】
若要使映射几何中心重合，则令：
(M-1)/2+Z=((N-1)/2+Z)*(M/N)
得Z=0.5
"""



# 直方图均衡化
def hg_equalization(img):
    h, w, c = img.shape
    empty_img = np.zeros((h, w, c), np.uint8)
    for i in range(c):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        sum_N = np.cumsum(hist)
        sum_Pi = sum_N / (h * w)
        sh = np.uint8(sum_Pi * 256 - 0.5)
        for x in range(h):
            for y in range(w):
                empty_img[x, y, i] = sh[img[x, y, i]]
    return empty_img



img = cv2.imread("lenna.png")
out0 = ne_interpolate(img, 500, 500)
out1 = bl_interpolate(img, 800, 800)
out2 = hg_equalization(img)
cv2.imshow("result0", out0)
cv2.imshow("result1", out1)
cv2.imshow("result2", out2)
cv2.waitKey(0)
