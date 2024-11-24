import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    emptyImg = np.zeros((800, 800, channels), np.uint8)  # 800*800像素，与输入图像通道数相同
    # sh = 800 / height
    # sw = 800 / width
    # for i in range(800):
    #     for j in range(800):
    #         x = int(i / sh + 0.5)
    #         y = int(j / sw + 0.5)
    #         emptyImg[i, j] = img[x, y]

    sh = height / 800
    sw = width / 800
    # 遍历新图像的每个像素点，计算对应位置
    for i in range(800):
        for j in range(800):
            x = int(i * sh + 0.5)
            y = int(j * sw + 0.5)
            emptyImg[i, j] = img[x, y]      #将原始图像 img 中坐标为 (x, y) 的像素值复制到新图像 emptyImg 的坐标为 (i, j) 的位置上。
    return emptyImg


img = cv2.imread("F:\DeepLearning\Code_test\lenna.png")
NearImg = function(img)
print(NearImg)
print(NearImg.shape)
cv2.imshow("NearestImg", NearImg)
cv2.imshow("OriginalImg", img)
cv2.waitKey(0)      # 暂停程序运行，不然图片就会一闪而过，等待按键才会继续执行代码
