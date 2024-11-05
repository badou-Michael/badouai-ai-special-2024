# -*- coding: utf-8 -*-
# time: 2024/10/26 16:04
# file: Canny.py
# author: flame
import cv2
import numpy as np


if __name__ == '__main__':
    # 读取图像文件，此处假设当前目录下有名为'lenna.png'的图像文件
    # 注意：cv2.imread函数读取图像后，默认的颜色空间是BGR，而非RGB
    img = cv2.imread("lenna.png")

    # 将读取的BGR图像转换为灰度图像
    # 这是因为在进行Canny边缘检测之前，需要确保图像是灰度的
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 显示灰度图像
    cv2.imshow("img_gray",img_gray)

    # 对灰度图像应用Canny边缘检测算法
    # 参数100和150分别是Canny算法的两个阈值，用于判定像素是否为边缘
    # 较低的阈值用于检测较弱的边缘，而较高的阈值用于检测较强的边缘
    cv2.imshow("canny",cv2.Canny(img_gray,100,150))

    # 等待任意按键按下后继续执行后续代码
    # 此处的作用是确保显示的图像窗口能够停留，以便用户可以观察到处理结果
    cv2.waitKey()

    # 关闭所有由cv2.imshow创建的窗口
    # 这是为了释放资源，避免内存泄漏
    cv2.destroyAllWindows()