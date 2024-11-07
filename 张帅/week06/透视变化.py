import cv2
import numpy as np

img = cv2.imread('E:\lenna.jpg')
y,x = img.shape[:2]

src_img = np.float32([[0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0], [0, 0]])

dst_img = np.float32([[0, int(img.shape[0]/2)], [int(img.shape[1]/2), int(img.shape[0]/2)], [int(img.shape[1]/2), 0], [0, 0]])
print(src_img)
print(dst_img)
# 计算透视变换矩阵
matrix_array = cv2.getPerspectiveTransform(src_img, dst_img)
# 透视变换
transformed_img = cv2.warpPerspective(img, matrix_array, (int(img.shape[1]/2), int(img.shape[0]/2)))

cv2.imshow("original image", img)
cv2.imshow("transformed image", transformed_img)
cv2.waitKey(0)
