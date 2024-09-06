import cv2

# 读原始图
img_org = cv2.imread("lenna.png")
cv2.imshow("original image", img_org)

# 灰度化
img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray level image", img_gray)

# print(img_gray)

# 二值化
_, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("binary image", img_binary)

cv2.waitKey(0)
