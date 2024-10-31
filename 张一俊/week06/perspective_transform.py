import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('photo1.jpg')
# 查找坐标
# plt.imshow(image)  # 鼠标放置定点，显示x=,y=
# plt.show()

# 定义源点和目标点
src_points = np.float32([[206, 152], [519, 286], [15, 603], [342, 732]])
dst_points = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# 应用透视变换
result = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Perspective Transform', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
