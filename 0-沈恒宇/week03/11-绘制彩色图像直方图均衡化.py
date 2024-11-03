import cv2

# 彩色图像直方图均衡化
img = cv2.imread("lenna.png")
cv2.imshow("img", img)

# 彩色图像均衡化
(b, r, g) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
