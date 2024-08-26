"""
双线性插值
"""
import cv2
import numpy as np


def process(grayImg, distWeight, distHeight):
    srcWeight, srcHeight = grayImg.shape[:2]
    if (srcWeight == distWeight and srcHeight == distHeight):
        return grayImg
    distImg = np.zeros([distWeight, distHeight, grayImg.shape[2]], dtype=grayImg.dtype)
    scaleX = srcWeight / distWeight
    scaleY = srcHeight / distHeight
    print(scaleX, scaleY, distImg.shape)
    for channel in range(3):
        for distY in range(distHeight):
            for distX in range(distWeight):
                srcX = int(distX * scaleX)
                srcY = int(distY * scaleY)
                srcX1 = int(max(np.floor(srcX - 1), 0))
                srcX2 = int(min(srcX1 + 1, srcWeight - 1))
                srcY1 = int(max(np.floor(srcY - 1), 0))
                srcY2 = int(min(srcY1 + 1, srcHeight - 1))
                temp0 = (srcX2 - srcX) * grayImg[srcX1, srcY1, channel] + (srcX - srcX1) * grayImg[
                    srcX2, srcY1, channel]
                temp1 = (srcX2 - srcX) * grayImg[srcX1, srcY2, channel] + (srcX - srcX1) * grayImg[
                    srcX2, srcY2, channel]
                distImg[distX, distY, channel] = (srcY2 - srcY) * temp0 + (srcY - srcY1) * temp1
    return distImg


if __name__ == '__main__':
    distWeight = int(input("请输入目标的宽度:"))
    distHeight = int(input("请输入目标的高度:"))
    grayImg = cv2.imread("lenna.png")
    distImg = process(grayImg, distWeight, distHeight)
    print(grayImg.shape, distImg.shape)
    cv2.imshow("grayImg", grayImg)
    cv2.imshow("distImg", distImg)
    cv2.waitKey(0)
