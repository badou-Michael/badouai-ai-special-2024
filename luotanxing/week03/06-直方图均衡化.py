import cv2
from matplotlib import pyplot as plt

# 获取灰度图像
img=cv2.imread('../week02/lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
heqResult = cv2.equalizeHist(img_gray);
cv2.imshow("equalize_img", heqResult);
cv2.imshow("img_gary", img_gray)


plt.figure()# 新建一个新的图形窗口
plt.title("Fimg_gary Histogram")
plt.hist(img_gray.ravel(), 256)# 0-256的直方图


plt.figure()# 新建一个新的图形窗口
plt.title("equalize Histogram")
plt.hist(heqResult.ravel(), 256)# 0-256的直方图
plt.show()

cv2.waitKey(0);