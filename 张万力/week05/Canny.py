import cv2
import numpy as np


'''
1.对图像进行灰度化
2.对图像进行高斯滤波
3.检测图像中的水平、垂直和对角边缘
4.对梯度幅值进行非极大值抑制
5.用双阈值算法检测和连接边缘
'''


def gaussian_blur(img, kernel_size=5):
    """高斯模糊,通过平滑操作降低图像噪声"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def sobel_filters(img):
    """计算x和y方向的Sobel梯度"""
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(Gx,Gy)#计算梯度幅值

    angle = np.arctan2(Gy, Gx)#计算梯度方向
    return magnitude, angle


def non_max_suppression(magnitude, angle):
    """非极大值抑制"""
    M, N = magnitude.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = angle * 180.0 / np.pi #将梯度方向转换为角度
    angle[angle < 0] += 180 #调整负角度到正范围内

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # 检查四个方向
            try:
                q = 255
                r = 255

                # 角度量化到四个方向之一【根据梯度方向，对比相邻像素的梯度幅值】
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                #如果当前点的梯度幅值是最大的，保留该点
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    Z[i, j] = magnitude[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThreshold, highThreshold):
    """应用双阈值"""
    highThreshold = img.max() * highThreshold #高阈值
    lowThreshold = highThreshold * lowThreshold #低阈值

    print("lower_boundary", lowThreshold);
    print("high_boundary", highThreshold)

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32) #初始化全0的数组

    weak = np.int32(25) #弱边缘标志
    strong = np.int32(255) #强边缘标志

    strong_i, strong_j = np.where(img >= highThreshold) # 强边缘的位置
    zeros_i, zeros_j = np.where(img < lowThreshold) # 小于低阈值的不是边缘

    weak_i, weak_j = np.where((img >= lowThreshold) & (img < highThreshold)) # 介于中间的是弱边缘的位置

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    res[zeros_i, zeros_j] = 0  # 明确设置低于低阈值的像素为0，虽然它们已经是0

    return res


def hysteresis(img, weak=25, strong=255):
    """滞后阈值处理进行边缘跟踪"""
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):  # 如果是弱边缘
                # 查看周围是否有强边缘
                if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong) or
                        (img[i, j - 1] == strong) or (img[i, j + 1] == strong) or
                        (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                    img[i, j] = strong # 将弱边缘升级为强边缘
                else:
                    img[i, j] = 0 # 否则去除该边缘
    return img


def canny_edge_detection(image, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# 将彩色图片灰度化
    blurred_image = gaussian_blur(gray)# 应用高斯模糊->降噪
    gradient_magnitude, gradient_angle = sobel_filters(blurred_image) # 计算梯度幅度和角度
    non_max_img = non_max_suppression(gradient_magnitude, gradient_angle) # 非极大值抑制
    threshold_img = threshold(non_max_img, lowThresholdRatio, highThresholdRatio)# 双阈值处理
    final_img = hysteresis(threshold_img)# 滞后阈值处理
    return final_img


if __name__ == "__main__":
    image = cv2.imread('../lenna.png')
    edge_detected_image = canny_edge_detection(image)

    cv2.imshow("canny Image", edge_detected_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
