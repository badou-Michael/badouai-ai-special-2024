#最临近插值
import cv2
import numpy as np

def nearest_neighbor_interpolation(image, scale):
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)

    new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            x = int(i / scale)
            y = int(j / scale)
            new_image[i, j] = image[x, y]

    return new_image

image = cv2.imread('input.jpg')
scaled_image = nearest_neighbor_interpolation(image, scale=2)
cv2.imwrite('output_nearest.jpg', scaled_image)

#双线性插值
def bilinear_interpolation(image, scale):
    """
    双线性插值
    :param image: 输入图像
    :param scale: 缩放比例
    :return: 缩放后的图像
    """
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)

    # 创建一个新的图像
    new_image = np.zeros((new_height, new_width, image.shape[2]), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # 计算原始图像中的对应位置
            x = i / scale
            y = j / scale

            # 计算四个最近的像素点
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, height - 1)
            y2 = min(y1 + 1, width - 1)

            # 计算权重
            dx = x - x1
            dy = y - y1

            # 双线性插值
            new_image[i, j] = (1 - dx) * (1 - dy) * image[x1, y1] + \
                              dx * (1 - dy) * image[x2, y1] + \
                              (1 - dx) * dy * image[x1, y2] + \
                              dx * dy * image[x2, y2]

    return new_image


# 示例使用
image = cv2.imread('input.jpg')
scaled_image = bilinear_interpolation(image, scale=2)
cv2.imwrite('output_bilinear.jpg', scaled_image)

#证明中心重合+0.5
def center_alignment_proof():
    """
    证明中心重合+0.5
    """
    # 假设有一个3x3的图像
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.uint8)

    # 计算中心位置
    center_x = image.shape[0] // 2
    center_y = image.shape[1] // 2

    # 打印中心位置
    print(f"中心位置: ({center_x}, {center_y})")

    # 计算中心位置加上0.5
    center_x_plus_0_5 = center_x + 0.5
    center_y_plus_0_5 = center_y + 0.5

    # 打印中心位置加上0.5
    print(f"中心位置+0.5: ({center_x_plus_0_5}, {center_y_plus_0_5})")


# 示例使用
center_alignment_proof()



#直方图均衡化

def histogram_equalization(image):
    """
    直方图均衡化
    :param image: 输入图像
    :return: 均衡化后的图像
    """
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 直方图均衡化
    equalized_image = cv2.equalizeHist(gray_image)

    return equalized_image

# 示例使用
image = cv2.imread('input.jpg')
equalized_image = histogram_equalization(image)
cv2.imwrite('output_equalized.jpg', equalized_image)
