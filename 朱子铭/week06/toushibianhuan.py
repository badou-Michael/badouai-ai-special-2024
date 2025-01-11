"""
    6.1toushibianhuan
"""
import cv2
import numpy as np
# 导入 OpenCV 和 NumPy 库，用于图像处理和数值计算。

img = cv2.imread("1.jpg")
rows, cols = img.shape[:2]
# 读取名为 "1.jpg" 的图像，并获取图像的行数（rows）和列数（cols），只取前两个维度（高度和宽度）。

src_points = np.float32([[260,149],[683,177],[836,620],[322,650]])
dst_points = np.float32([[0,0],[520,0],[520,520],[0,520]])
# 定义原始图像中的四个点（src_points）和目标图像中对应的四个点（dst_points），坐标为浮点数类型。

M = cv2.getPerspectiveTransform(src_points,dst_points)
# 使用 OpenCV 的函数 getPerspectiveTransform 计算从原始点到目标点的透视变换矩阵 M。

dst = cv2.warpPerspective(img, M, (520,520))
# 使用透视变换矩阵 M 对原始图像 img 进行透视变换，得到变换后的图像 dst，目标图像大小为 (520,520)。

print(M)
# 打印透视变换矩阵 M。

cv2.imshow('Original Image', img)
cv2.imshow('Transformed Image', dst)
# 显示原始图像和变换后的图像，窗口标题分别为 'Original Image' 和 'Transformed Image'。

cv2.waitKey(0)
cv2.destroyAllWindows()
# 等待用户按键，当用户按下任意键时，程序继续执行。
# 关闭所有打开的图像窗口。
