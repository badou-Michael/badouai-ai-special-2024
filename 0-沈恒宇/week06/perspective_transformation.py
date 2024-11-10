"""利用透视变换的原理，获取图片中的纸片的图片"""
import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
temp = img.copy()
"""在计算机科学和编程中，src 和 dst 是常用的缩写，通常表示 "source"（源）和 "destination"（目标）。
它们用于描述数据或信息的起点和终点。
src：源，表示数据的来源或起始位置。
dst：目标，表示数据的目的地或结束位置。"""
src = np.float32([[207, 151], [517, 285], [17,601], [343,731]])
dst = np.float32([[0,0],[300,0],[0,485],[300,485]])
# 生成透视变换矩阵，进行透视变换
M = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix",M)
result = cv2.warpPerspective(temp, M, (300,485))
cv2.imshow('src',img)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
