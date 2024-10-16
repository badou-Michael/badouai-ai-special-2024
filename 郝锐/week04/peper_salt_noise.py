import cv2
import random
import numpy as np
def peper_salt_noise(src, salt_prob, pepper_prob):
    """
    Add salt and pepper noise to an image
    :param src: 源图像
    :param salt_prob:盐噪声出现的概率 0-1
    :param pepper_prob:椒噪声出现的概率0-1
    :return: 添加椒盐噪声后的图像
    """
    # 获取图像的宽度和高度
    height, width = src.shape[:2]

    # 创建一个与源图像大小相同的空白图像
    dst = src.copy()
    # 计算盐噪声的数量,向上取整
    salt_num = np.ceil(src.size * salt_prob)
    # 计算椒噪声的数量,向上取整
    pepper_num = np.ceil(src.size * pepper_prob)
    # 遍历图像的每个像素,边缘不做处理
    for i in range(int(salt_num)):
        # 随机生成一个像素位置
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        dst[x, y] = 255  # 添加盐噪声
    for i in range(int(pepper_num)):
        # 随机生成一个像素位置
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        dst[x, y] = 0  # 添加椒噪声
    return dst
if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)  # 读取灰度图像
    dst = peper_salt_noise(img, 0.01, 0.01)  # 添加椒盐噪声
    cv2.imshow('source', img)  # 显示源图像
    cv2.imshow('peper_salt', dst)  # 显示添加椒盐噪声后的图像
    cv2.waitKey(0)
