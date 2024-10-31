import cv2
import numpy as np

img = cv2.imread('1.png')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[112, 209], [447, 171], [216, 496], [544, 363]])
dst = np.float32([[0, 0], [335, 0], [0, 488], [335, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (335, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
