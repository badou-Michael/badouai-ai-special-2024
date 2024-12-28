import numpy as np
import cv2

def compute_perspective_transform(src_points, dst_points):
    # 构建A矩阵
    A = []
    for i in range(4):
        x, y = src_points[i]
        x_prime, y_prime = dst_points[i]
        A.append([x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime])
        A.append([0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime])
    A = np.array(A)
    
    # 构建b向量
    b = np.array([dst_points[0][0], dst_points[0][1],
                  dst_points[1][0], dst_points[1][1],
                  dst_points[2][0], dst_points[2][1],
                  dst_points[3][0], dst_points[3][1]])
    
    # 求解线性方程组
    h = np.linalg.solve(A, b)
    
    # 构建透视变换矩阵
    H = np.array([[h[0], h[1], h[2]],
                  [h[3], h[4], h[5]],
                  [h[6], h[7], 1]])
    
    return H

def apply_perspective_transform(image, H, output_shape):
    # 使用cv2.warpPerspective函数来应用变换矩阵
    transformed_image = cv2.warpPerspective(image, H, output_shape)
    return transformed_image

# 示例使用
if __name__ == "__main__":
    # 加载图像
    img = cv2.imread('photo1.jpg')
    
    if img is None:
        raise ValueError("Image not found or unable to read")

    # 定义源图像和目标图像中的四个点
    src_points = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst_points = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    
    # 计算透视变换矩阵
    H = compute_perspective_transform(src_points, dst_points)
    
    # 应用透视变换
    result = apply_perspective_transform(img, H, (350, 488))
    
    # 显示结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Transformed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()