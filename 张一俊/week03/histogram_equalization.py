import cv2
import matplotlib.pyplot as plt

'''
直方图均衡化一：灰度图：1.准备图像 2.均衡化cv2.equalizeHist 3.绘制直方图
'''
original_img = cv2.imread("lenna.png")
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray img", gray_img)

equal_gray_img = cv2.equalizeHist(gray_img)
cv2.imshow("equal gray img", equal_gray_img)
cv2.waitKey(0)

equal_hist = cv2.calcHist([equal_gray_img], [0], None, [256], [0, 256])

plt.figure()
plt.plot(equal_hist)
plt.show()

# cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))

'''
直方图均衡化二：彩色图：1.准备图像 2.均衡化cv2.equalizeHist 3.绘制直方图
'''
original_img = cv2.imread("lenna.png")
cv2.imshow("original img", original_img)
(blue, green, red) = cv2.split(original_img)

blue_equal_hist = cv2.equalizeHist(blue)
green_equal_hist = cv2.equalizeHist(green)
red_equal_hist = cv2.equalizeHist(red)

# 合并
equal_img = cv2.merge((blue_equal_hist, green_equal_hist, red_equal_hist))
cv2.imshow("equal img", equal_img)
cv2.waitKey(0)
