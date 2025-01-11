import cv2
import numpy as np


def resize_function(image):
    # 读取原图，并获取原图的宽，长，通道
    height, width, channels = image.shape
    # 创建一张空白的图像，设置尺寸800*800，通道数与原图相同
    empty_image = np.zeros((800, 800, channels), np.uint8)
    # 计算缩放倍数
    sh = 800/height
    sw = 800/width
    # 双循环遍历新图像的每一个像素点（i,j）
    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)  # 像素值只能是整数，实现四舍五入的效果，更加准确
            y = int(j/sw + 0.5)
            # 将原图中的像素值赋个新图像
            empty_image[i, j] = image[x, y]
    return empty_image


# cv2,resize(img,(800,800,c),near/bin)

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    if img is not None:
        zoom = resize_function(img)
        cv2.imshow("Zoomed Image", zoom)  # 显示放大后的图像
        cv2.waitKey(0)  # 等待按键
    else:
        print("Error")
