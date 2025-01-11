import cv2
import numpy as np
from matplotlib import pyplot as plt

def Histogram_Equalize(img, img_type = 1):
    # input: - image
    #        - image type to be histogram equalized
    if img_type == 1:
        # 灰度图像直方图均衡化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 直方图均衡化
        img_equalHist = cv2.equalizeHist(gray)

        hist = cv2.calcHist([img_equalHist],[0],None,[256],[0,256])
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")#X轴标签
        plt.ylabel("# of Pixels")#Y轴标签
        plt.plot(hist)
        plt.show()

        cv2.imshow("Histogram Equalization", np.hstack([gray, img_equalHist]))
        cv2.waitKey(0)
    elif img_type == 2:
        # 彩色直方图均衡化

        (b, g, r) = cv2.split(img)
        b_H = cv2.equalizeHist(b)
        g_H = cv2.equalizeHist(g)
        r_H = cv2.equalizeHist(r)
        img_equalHist = cv2.merge((b_H, g_H, r_H))
        cv2.imshow("Histogram Equalization", np.hstack([img, img_equalHist]))
        cv2.waitKey(0)

if __name__ == '__main__':
    img = cv2.imread("lenna.png", 1)

    Histogram_Equalize(img, 1)