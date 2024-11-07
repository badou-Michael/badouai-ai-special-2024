import cv2
import numpy as np

# 读取图像
image = cv2.imread('photo1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊，减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 进行边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到面积最大的轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)
paper_contour = contours[0]

# 近似四边形的轮廓
epsilon = 0.02 * cv2.arcLength(paper_contour, True)
approx = cv2.approxPolyDP(paper_contour, epsilon, True)

# 如果找到的轮廓是四个点
if len(approx) == 4:
    points_original = np.float32([point[0] for point in approx])

    # 定义目标图像的四个顶点
    width, height = 500, 700
    points_target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 计算透视变换矩阵并应用变换
    matrix = cv2.getPerspectiveTransform(points_original, points_target)
    result = cv2.warpPerspective(image, matrix, (width, height))

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Transformed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("未能找到纸张的四个角点")
