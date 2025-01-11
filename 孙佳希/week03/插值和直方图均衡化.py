import cv2
import numpy as np
import matplotlib.pyplot as plt

#最邻近插值
# def nearest(img):
#     h, w, c = img.shape
#     empty_img = np.zeros((800, 800, c), np.uint8)
#     h_bili = 800 / h
#     w_bili = 800 / w
#     for i in range(800):
#         for j in range(800):
#             x = int(i / h_bili + 0.5)
#             y = int(j / w_bili + 0.5)
#             empty_img[i, j] = img[x, y]
#
#     return empty_img
#
#
# image = cv2.imread("lenna.png")
# big_image = nearest(image)
# print(image.shape)
# print(big_image.shape)
# cv2.imshow("1", image)
# cv2.waitKey(0) #防止图像自动关闭


# #双线性插值
# def bilinear_interpolation(img, out_dim):
#     h, w, c = img.shape
#     tarH, tarW = out_dim[1], out_dim[0]
#     if h == tarH and w == tarW:
#         return img.copy()
#     tarImg = np.zeros((tarH, tarW, 3), dtype=np.uint8)
#     h_bili = float(h/tarH)
#     w_bili = float(w/tarW)
#     for n in range(c):
#         for i in range(tarH):
#             for j in range(tarW):
#                 src_x = (j+0.5)*w_bili - 0.5
#                 src_y = (i+0.5)*h_bili - 0.5
#
#                 src_x0 = int(np.floor(src_x))
#                 src_x1 = min(src_x0 + 1, w - 1)
#                 src_y0 = int(np.floor(src_y))
#                 src_y1 = min(src_y0 + 1, h - 1)
#
#                 temp0 = (src_x1 - src_x) * img[src_y0, src_x0, n] + (src_x - src_x0) * img[src_y0, src_x1, n]
#                 temp1 = (src_x1 - src_x) * img[src_y1, src_x0, n] + (src_x - src_x0) * img[src_y1, src_x1, n]
#                 tarImg[i, j, n] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
#     return tarImg
#
#
# if __name__ == '__main__':
#     img = cv2.imread('lenna.png')
#     dst = bilinear_interpolation(img, (700, 700))
#     cv2.imshow('bilinear interp', dst)
#     cv2.waitKey()
#
#
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''

# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#直方图均衡化
dst = cv2.equalizeHist(gray)

dst_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()
#
# cv2.imshow("histogram equalization", np.hstack([gray,dst]))
# cv2.waitKey(0)

#彩色图像均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 显示彩色图像直方图
plt.figure(figsize=(10, 5))
plt.hist(r.ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
plt.hist(g.ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
plt.hist(b.ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
plt.show()

#显示均衡化后的rgb图像
res = cv2.merge((bH, gH, rH))
cv2.imshow("color Hist", np.hstack([img, res]))
cv2.waitKey(0)
