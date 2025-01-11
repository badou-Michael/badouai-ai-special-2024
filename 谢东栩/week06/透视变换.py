import cv2
import numpy as np

img = cv2.imread('cs.jpg')

#保留一下副本
result3 = img.copy()

#目标坐标
src = np.float32([[150, 120], [4500, 2050], [10, 500], [3000, 7000]])
dst = np.float32([[0, 0], [400, 0], [0, 488], [400, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
#输出透视矩阵，a33默认为0
print(m)
#进行图像透视转换，参数要和dst中的矩阵匹配
result = cv2.warpPerspective(result3, m, (400, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
