import cv2

# 读取灰度图
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 均衡化
equalized_img = cv2.equalizeHist(img)

# 显示图像
cv2.imshow('Equalized Image', equalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
