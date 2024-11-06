import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()

# 定义源坐标和目标坐标 找出四个角点
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(result3, m, (512, 512))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
