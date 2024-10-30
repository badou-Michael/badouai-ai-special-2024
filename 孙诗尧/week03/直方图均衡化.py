import cv2
img = cv2.imread('lenna.png')
b, g, r = cv2.split(img)
# 对每个通道进行直方图均衡化
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并通道
img_equalized = cv2.merge([bH, gH, rH])
cv2.imshow("Lenna", img)
cv2.imshow("Lenna after histogram equalization", img_equalized)
cv2.waitKey(0)
