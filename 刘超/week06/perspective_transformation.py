import numpy as np
import cv2

def compute_matrix(src: np.array, dst: np.array) -> np.ndarray:
    A = []
    B = []
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.extend([u, v])
    A, B = np.array(A), np.array(B)
    H = np.linalg.solve(A, B)
    H = np.append(H, 1).reshape(3, 3)
    return H

def apply_transform(image: np.ndarray, H: np.ndarray, output_size) -> np.ndarray:
    src_height, src_width = image.shape[:2]
    dst_width, dst_height = output_size
    dst_image = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    for y in range(dst_height):
        for x in range(dst_width):
            inv_H = np.linalg.inv(H)
            dst_point = np.array([x, y, 1])
            src_point = np.dot(inv_H, dst_point)
            src_point /= src_point[2]

            x0, y0 = int(src_point[0]), int(src_point[1])
            x1, y1 = x0 + 1, y0 + 1
            if 0 <= x0 < src_width - 1 and 0 <= y0 < src_height - 1:
                dx = src_point[0] - x0
                dy = src_point[1] - y0
                p00 = image[y0, x0]
                p10 = image[y0, x1]
                p01 = image[y1, x0]
                p11 = image[y1, x1]
                pixel_value = (
                    (1 - dx) * (1 - dy) * p00 +
                    dx * (1 - dy) * p10 +
                    (1 - dx) * dy * p01 +
                    dx * dy * p11
                )
                dst_image[y, x] = pixel_value.astype(np.uint8)
    return dst_image

src_points = np.array([[50, 50], [250, 50], [250, 250], [50, 250]])
dst_points = np.array([[0, 0], [300, 0], [300, 300], [0, 300]])

# 使用 cv2 读取图像
image = cv2.imread("lenna.png")

# 计算透视变换矩阵
H = compute_matrix(src_points, dst_points)

# 应用透视变换
output_image = apply_transform(image, H, (600, 400))

# 使用 cv2 存储图像
cv2.imwrite("output_image2.png", output_image)

pts_src = np.float32([[100, 100], [500, 100], [100, 400], [500, 400]])
pts_dst = np.float32([[150, 150], [450, 150], [150, 350], [450, 350]])

# 读取源图像
img_src = cv2.imread('lenna.png')

# 计算变换矩阵
H, _ = cv2.findHomography(pts_src, pts_dst)

# 应用透视变换
width, height = 600, 400  # 目标图像的尺寸
img_dst = cv2.warpPerspective(img_src, H, (width, height))

# 显示结果

cv2.imshow('Source Image', img_src)
cv2.imshow('Warped Image', img_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
