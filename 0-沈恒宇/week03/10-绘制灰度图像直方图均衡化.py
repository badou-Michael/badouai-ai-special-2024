import cv2
from matplotlib import pyplot as plt

# img,img_gray,dst都是二维数组，包含了像素值
# 获取灰度图像
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gary", img_gray)
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(img_gray)
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("new_gray", dst)
cv2.waitKey(0)

