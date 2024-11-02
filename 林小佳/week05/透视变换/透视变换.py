import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
result_ = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])  # 变换后图像尺寸为337*488
# 生成透视矩阵warpMatrix
m = cv2.getPerspectiveTransform(src, dst)
# 进行透视变换
result = cv2.warpPerspective(result_, m, (337, 488))

cv2.imshow("src", img)
cv2.imshow("dst", result)
cv2.waitKey()
