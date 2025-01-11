import matplotlib.pyplot as plt
import cv2

# 灰度直方图均衡化
img = cv2.imread("../lenna.png")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow("CV2Gray", imgGray)
# cv2.waitKey(0)
# 对ImgGray做直方图均衡化处理，使其灰度分布更均匀
dst = cv2.equalizeHist(imgGray)
# 计算每个灰度像素数量
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
# plt展示
plt.subplot(121)
plt.title("imgGrayHist")
plt.hist(imgGray.ravel(), 256)

plt.subplot(122)
plt.title("imgGrayEqualizeHist")
plt.hist(dst.ravel(), 256)
plt.show()

# 彩色直方图均衡化
img_2 = cv2.imread("../lenna.png")
# 分割三个通道
(b, g, r) = cv2.split(img_2)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并三个通道
dst_result = cv2.merge((bH, gH, rH))
# 展示结果
cv2.imshow("colorImg", img_2)
cv2.imshow("colorImgHist", dst_result)
cv2.waitKey(0)
