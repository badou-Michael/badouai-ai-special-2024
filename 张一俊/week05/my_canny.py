import numpy as np
from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import cv2

# 高斯滤波
def gaussian_blur(gray_image, kernel_size=5, sigma=0.5):
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), sigma)


def gradient_magnitude_and_direction(image):
    # Sobel算子计算梯度
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值和方向
    magnitude = np.hypot(sobel_x, sobel_y)
    direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi) % 180

    return magnitude, direction


def non_maximum_suppression(magnitude, direction):
    height, width = magnitude.shape
    output = np.zeros((height, width), dtype=np.float32)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = direction[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i, j + 1], magnitude[i, j - 1]]
            elif (22.5 <= angle < 67.5):
                neighbors = [magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]]
            elif (67.5 <= angle < 112.5):
                neighbors = [magnitude[i + 1, j], magnitude[i - 1, j]]
            else:
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

            if magnitude[i, j] >= max(neighbors):
                output[i, j] = magnitude[i, j]

    return output


def double_threshold(image, low_threshold, high_threshold):
    strong = 255
    weak = 0

    output = np.zeros_like(image)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak

    return output


def edge_tracking_by_hysteresis(image):
    height, width = image.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if image[i, j] == 75:  # Weak edge
                if ((image[i + 1, j - 1] == 255) or (image[i + 1, j] == 255) or (image[i + 1, j + 1] == 255) or
                    (image[i, j - 1] == 255) or (image[i, j + 1] == 255) or
                    (image[i - 1, j - 1] == 255) or (image[i - 1, j] == 255) or
                    (image[i - 1, j + 1] == 255)):
                    image[i, j] = 255
                else:
                    image[i, j] = 0
    return image


# 法一：手写实现
def canny_principle(gray_image, low_threshold, high_threshold):
    # 0.灰度化(已实现)
    # gray_img = cv2.imread("lenna.png", flags = 0)
    # or
    # img = cv2.imread("lenna.png")
    # gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1.高斯平滑
    blurred_image = gaussian_blur(gray_image)
    # return bluured_image

    # 2.计算梯度和方向:用Soble算子检测水平&垂直&对角边缘
    magnitude, direction = gradient_magnitude_and_direction(blurred_image)

    # 3.非极大值抑制
    non_max_image = non_maximum_suppression(magnitude, direction)

    # 4.双阈值检测
    threshold_image = double_threshold(non_max_image, low_threshold, high_threshold)

    # 5.边缘连接
    final_edges = edge_tracking_by_hysteresis(threshold_image)

    return final_edges


# 法二：调用cv2的Canny接口
def canny_interface(gray, thshld1, thshld2):
    return cv2.Canny(gray, threshold1=thshld1, threshold2=thshld2)


def nothing(X):
# def nothing(): TypeError: nothing() takes 0 positional arguments but 1 was given
    pass

# 滑动条实现
def canny_interface_track(gray, thshld1, thshld2):
    # 回调函数，处理滑动条变化（可选）

    # 创建一个窗口
    cv2.namedWindow('Canny Edge Detection')

    # 创建滑动条，参数为：滑动条名称、位置:所在窗口名称、初始值、最大值、回调函数(滑动条变化时使用)
    cv2.createTrackbar('Threshold1', 'Canny Edge Detection', thshld1, 255, nothing)
    cv2.createTrackbar('Threshold2', 'Canny Edge Detection', thshld2, 255, nothing)

    # 读取图像
    gray_image = cv2.imread('lenna.png', flags=0)

    while True:
        # 获取滑动条的当前值/位置position
        threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny Edge Detection')
        threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny Edge Detection')

        # 根据滑动条的值调整图像（例如：简单地改变亮度），
        # 将输入图像进行缩放（scale）和取绝对值（abs），处理图像亮度或对比度常用方法
        adjusted_image = cv2.Canny(gray_image, threshold1=threshold1, threshold2=threshold2)

        # 显示调整后的图像
        cv2.imshow('Image', adjusted_image)

        # 按 'q' 键退出循环
        # waitKey(0) 用于暂停程序，等待用户输入； waitKey(1) 等待1ms,用于快速循环显示图像
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cv2.destroyAllWindows()


if __name__ == "__main__":

    gray = cv2.imread("lenna.png", flags=0)
    cv2.imshow("original_gray", gray)

    thshld1 = 100
    thshld2 = 200

    canny_1 = canny_principle(gray, thshld1, thshld2)
    cv2.imshow("canny principle", canny_1)

    canny_2 = canny_interface(gray, thshld1, thshld2)
    cv2.imshow("canny of cv2", canny_2)

    canny_3 = canny_interface_track(gray, thshld1, thshld2)

    cv2.waitKey()


