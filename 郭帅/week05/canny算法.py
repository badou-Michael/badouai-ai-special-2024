import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_edge_detection(img, low_ratio, high_ratio):
    # 转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 步骤 1: 高斯平滑
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0.5)

    # 步骤 2: 计算梯度（使用Sobel算子）
    grad_x = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)

    # 步骤 3: 计算梯度幅值和方向
    magnitude = cv2.magnitude(grad_x, grad_y)
    direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)

    # 步骤 4: 非最大值抑制
    suppressed_img = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            angle = direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbor1 = magnitude[i, j + 1]
                neighbor2 = magnitude[i, j - 1]
            elif (22.5 <= angle < 67.5):
                neighbor1 = magnitude[i + 1, j - 1]
                neighbor2 = magnitude[i - 1, j + 1]
            elif (67.5 <= angle < 112.5):
                neighbor1 = magnitude[i + 1, j]
                neighbor2 = magnitude[i - 1, j]
            else:
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                suppressed_img[i, j] = magnitude[i, j]

    # 步骤 5: 双阈值处理
    low_threshold, high_threshold = np.percentile(suppressed_img, [low_ratio * 100, high_ratio * 100])
    strong_edges = (suppressed_img > high_threshold)
    weak_edges = ((suppressed_img >= low_threshold) & (suppressed_img <= high_threshold))

    # 步骤 6: 边缘连接
    final_edges = np.zeros_like(strong_edges, dtype=np.uint8)
    final_edges[strong_edges] = 255
    for i in range(1, suppressed_img.shape[0] - 1):
        for j in range(1, suppressed_img.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    final_edges[i, j] = 255

    return final_edges


# 加载图像
img = cv2.imread('lenna.png')

# 执行 Canny 边缘检测
edges = canny_edge_detection(img,0.6,0.9)

# 保存结果
cv2.imwrite("lenna5.png",edges)
