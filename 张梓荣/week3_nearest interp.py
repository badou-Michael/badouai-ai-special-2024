"""
最邻近插值The nearest interpolation
"""
import cv2
import numpy as np


def process(img, distWeight, distHeight):
    srcWidth, srcHeight, channels = img.shape
    if (srcWidth == distWeight and srcHeight == distHeight):
        return img
    distImg = np.zeros([distWeight, distHeight, channels], dtype=img.dtype)
    scaleHeight = srcHeight / distHeight
    scaleWidth = srcWidth / distWeight
    for distX in range(distHeight):
        for distY in range(distWeight):
            srcX = int(distX * scaleHeight)
            srcY = int(distY * scaleWidth)
            if srcX >= srcHeight:
                srcX = srcHeight - 1
            if srcY >= srcWidth:
                srcY = srcWidth - 1
            distImg[distX, distY] = img[srcX, srcY]
    return distImg


if __name__ == '__main__':
    distWeight = int(input("请输入目标的宽度:"))
    distHeight = int(input("请输入目标的高度:"))
    img = cv2.imread('lenna.png')
    distImg = process(img, distWeight, distHeight)
    # cv2.resize(img,)
    cv2.imshow("img", img)
    cv2.imshow("distImg", distImg)
    cv2.waitKey(0)
