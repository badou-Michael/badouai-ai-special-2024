import numpy as np
import cv2
# 最邻近插值
# def img_insertvalue(img,a,b):
#     h,w,c=img.shape
#     newimg = np.zeros((a,b,c),np.uint8)
#     sh = a/h
#     sw = b/w
#     for i in range(h):
#         for j in range(w):
#             newimg[int(i*sh+0.5)][int(j*sw+0.5)] = img[i][j]
#     return newimg
# img =  cv2.imread("lenna.png")
#
# result = img_insertvalue(img,700,700)
# cv2.imshow("image",result)
# cv2.waitKey()
#双线性插值
# def img_doubleinset(img,a,b):
#     h,w,c=img.shape
#     newimg = np.zeros((a,b,3),dtype=np.uint8)
#     sh = float(h/a)
#     sw = float(w/b)
#     for k in range(c):
#         for i in range(a):
#             for j in range(b):
#                 src_x = (j + 0.5) * sw - 0.5
#                 src_y = (i+ 0.5) * sh - 0.5
#                 src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
#                 src_x1 = min(src_x0 + 1, w - 1)
#                 src_y0 = int(np.floor(src_y))
#                 src_y1 = min(src_y0 + 1, h - 1)
#                 temp0 = (src_x1 - src_x) * img[src_y0, src_x0, k] + (src_x - src_x0) * img[src_y0, src_x1, k]
#                 temp1 = (src_x1 - src_x) * img[src_y1, src_x0, k] + (src_x - src_x0) * img[src_y1, src_x1, k]
#                 newimg[i, j, k] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
#     return newimg
# img = cv2.imread("lenna.png")
# result = img_doubleinset(img,800,800)
# cv2.imshow("image",result)
# cv2.waitKey()
#直方图
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
