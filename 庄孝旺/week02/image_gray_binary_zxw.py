from skimage.color import rgb2gray
import cv2

# 原图像
img = cv2.imread("lenna.png")
cv2.imshow("image origin",img)

# 灰度化
img_gray = rgb2gray(img)
cv2.imshow("image gray",img_gray)

# 二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        img_gray[i, j] = 0 if img_gray[i, j] <= 0.5 else 255
cv2.imshow("image binary",img_gray)
cv2.waitKey(0)
