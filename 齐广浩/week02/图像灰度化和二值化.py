"""
@author: BigBoy
内容：使用 OpenCV 实现彩色图像的二值化和灰度化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray


def OpenCV_Gray_Binary():
    # 读取彩色图像
    img = cv2.imread("lenna.png")

    # 显示原始BGR格式图像
    plt.subplot(221)
    plt.title("BGR")
    plt.imshow(img)

    # 将图像从BGR格式转换为RGB格式（因为matplotlib使用RGB格式显示图像）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 显示RGB格式图像
    plt.subplot(222)
    plt.title("RGB")
    plt.imshow(img)

    # 获取图像的高度和宽度
    h, w = img.shape[:2]

    # 创建一个用于存储灰度图像的空数组，大小与原图像相同
    img_gray = np.zeros([h, w], img.dtype)

    # 手动将彩色图像转为灰度图（使用浮点算法计算灰度值）
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = int(m[0] * 0.3 + m[1] * 0.59 + m[2] * 0.11)

    # 显示手动转换得到的灰度图像
    plt.subplot(223)
    plt.title("GrayRGB")
    plt.imshow(img_gray, cmap="gray")  # 使用灰度色彩映射

    # 使用OpenCV显示灰度图像
    cv2.imshow("CV2 grayImage", img_gray)
    cv2.waitKey(1000)

    # 手动灰度图的二值化
    img_gray_B = img_gray / 255.0  # 将灰度图像的像素值归一化为[0, 1]范围
    img_binary = np.where(img_gray_B >= 0.5, 1.0, 0.0)  # 使用阈值0.5进行二值化处理

    # 显示二值化后的图像
    plt.subplot(224)
    plt.title("Binary_img")
    plt.imshow(img_binary, cmap="gray")

    # 显示所有的子图
    plt.show()

def PIL_gary_binary():
    # 使用 PIL 读取图像并转换为 RGB 格式
    img = Image.open("lenna.png").convert("RGB")

    # 将 PIL 图像转换为 NumPy 数组
    img_np = np.array(img)

    # 显示 RGB 格式图像
    plt.subplot(221)
    plt.title("RGB (PIL)")
    plt.imshow(img_np)

    # 使用 skimage 库的 rgb2gray 函数将图像转换为灰度图
    img_gray = rgb2gray(img_np)  # 结果是浮点型灰度图
    print("PIL Mat", img_gray)
    # 显示灰度图像
    plt.subplot(222)
    plt.title("Gray (PIL)")
    plt.imshow(img_gray, cmap="gray")

    # 灰度图的二值化处理
    img_binary = np.where(img_gray >= 0.5, 1.0, 0.0)

    # 显示二值化图像
    plt.subplot(223)
    plt.title("Binary_img (PIL)")
    plt.imshow(img_binary, cmap="gray")

    # 显示所有的子图
    plt.show()


if __name__ == '__main__':
    OpenCV_Gray_Binary()
    PIL_gary_binary()
