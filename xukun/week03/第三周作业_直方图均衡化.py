import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("lenna.png")
#灰度直方图均衡化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_eq_img = cv2.equalizeHist(gray)


# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
plt.figure()
# 显示直方图   彩色直方图
plt.hist(result.ravel(), 256, [0, 256])

#灰色直方图
#plt.hist(gray_eq_img.ravel(), 256, [0, 256])


# plt.hist(bH.ravel(), 256, [0, 256])
# plt.hist(gH.ravel(), 256, [0, 256])
# plt.hist(rH.ravel(), 256, [0, 256])
plt.show()
cv2.imshow("dst_rgb", np.hstack([img,result]))
cv2.imshow("dst_gray", gray_eq_img)
cv2.waitKey(0)
