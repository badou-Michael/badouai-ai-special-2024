"""
双线性插值
"""
import cv2
import numpy as np


def process(img, distWeight, distHeight):
    srcWeight, srcHeight = img.shape[:2]
    if (srcWeight == distWeight and srcHeight == distHeight):
        return img.copy()
    distImg = np.zeros([distWeight, distHeight, img.shape[2]], dtype=img.dtype)
    scaleX = srcWeight / distWeight
    scaleY = srcHeight / distHeight
    print(scaleX, scaleY, distImg.shape)
    for channel in range(3):
        for distY in range(distHeight):
            for distX in range(distWeight):
                # (srcX,srcY)是原图的虚拟点  srcX1,srcY1,srcX2,srcY2 是邻近的点
                srcX = (distX + 0.5) * scaleX - 0.5
                srcY = (distY + 0.5) * scaleY - 0.5
                srcX1 = int(max(np.floor(srcX), 0))
                srcX2 = int(min(np.ceil(srcX), srcWeight - 1))
                srcY1 = int(max(np.floor(srcY), 0))
                srcY2 = int(min(np.ceil(srcY), srcHeight - 1))
                # x轴
                temp0 = (srcX2 - srcX) * img[srcX1, srcY1, channel] + (srcX - srcX1) * img[
                    srcX2, srcY1, channel]
                temp1 = (srcX2 - srcX) * img[srcX1, srcY2, channel] + (srcX - srcX1) * img[
                    srcX2, srcY2, channel]
                # y轴
                distImg[distX, distY, channel] = int((srcY2 - srcY) * temp0 + (srcY - srcY1) * temp1)
                # print(srcX, srcY, srcX1, srcY1, srcX2, srcY2, temp0, temp1, distImg[distX, distY, channel])
    return distImg


if __name__ == '__main__':
    distWeight = int(input("请输入目标的宽度:"))
    distHeight = int(input("请输入目标的高度:"))
    img = cv2.imread("lenna.png")
    distImg = process(img, distWeight, distHeight)
    print(img.shape, distImg.shape)
    cv2.imshow("img", img)
    cv2.imshow("distImg", distImg)
    cv2.waitKey(0)
