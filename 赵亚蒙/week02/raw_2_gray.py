import cv2
import numpy as np


def raw_2_gray(path):
    # 读取图像
    raw_img = cv2.imread(path)
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    # 获取图像的高和宽
    h, w = raw_img.shape[:2]

    # 创建一张和原图大小一样的单通道图像
    img_gray = np.zeros((h,w),dtype=np.uint8)

    # 遍历每一个像素并计算灰度值
    for i in range(h):
        for j in range(w):
            rgb_v = img[i, j]
            img_gray[i, j] = int(rgb_v[0]*0.3+rgb_v[1]*0.59+rgb_v[2]*0.11)

    # 打印图像数字矩阵
    print("Gray image matrix:")
    print(img_gray)

    # 显示灰度图像
    cv2.imshow("Grayscale Image", img_gray)
    cv2.waitKey(0)  # 等待按键输入
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

if __name__ == '__main__':
    raw_2_gray(path='./lenna.png')

