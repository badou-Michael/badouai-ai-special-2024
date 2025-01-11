import cv2
import numpy as np
# def get_points(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         points.append((x, y))
#         if len(points) == 4 :
#             cv2.destroyAllWindows()
#
#
# points = []
img = cv2.imread('123.jpg')
# cv2.imshow('Select Points', img)
# cv2.setMouseCallback('Select Points', get_points)
# cv2.waitKey(0)
#
# print('Select Points', points)
# 取到四个点坐标  Select Points [(562, 370), (764, 544), (70, 931), (264, 1109)]

result3 = img.copy()
'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[562, 370], [764, 544], [70, 931], [264, 1109]])
dst = np.float32([[0, 0], [350, 0], [0, 688], [350, 688]])

print(img.shape)

# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)

print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (450, 688))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)