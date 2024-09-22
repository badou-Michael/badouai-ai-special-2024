# -*- coding: utf-8 -*-
import cv2
import numpy as np

def bilinear(img, dstHeight, dstWidth):
    srcH, srcW, srcC = img.shape
    if srcH == dstHeight and srcW == dstWidth:
        return img
    dstImg = np.zeros((dstHeight, dstWidth, srcC),np.uint8)
    scaleX = float(srcH)/dstWidth
    scaleY = float(srcW)/dstHeight
    for c in range(srcC):
        for x in range(dstWidth):
            for y in range(dstHeight):
                srcX = (x + 0.5) * scaleX - 0.5
                srcY = (y + 0.5) * scaleY - 0.5
                # 4个角
                srcX0 = int(np.floor(srcX))
                srcX1 = min(srcX0 + 1, srcW - 1)
                srcY0 = int(np.floor(srcY))
                srcY1 = min(srcY0 + 1, srcH - 1)
                # 计算差值
                tmp0 = (srcX1 - srcX) * img[srcY0, srcX0, c] + (srcX - srcX0) * img[srcY0, srcX1, c]
                tmp1 = (srcX1 - srcX) * img[srcY1, srcX0, c] + (srcX - srcX0) * img[srcY1, srcX1, c]
                dstImg[y, x, c] = int((srcY1 - srcY) * tmp0 + (srcY - srcY0) * tmp1)
    return dstImg

if __name__ == '__main__':
    img = cv2.imread("../lenna.png")
    imgNew2 = bilinear(img, 800, 800)
    cv2.imshow('bilinear interp2', imgNew2)
    imgNew1 = bilinear(img, img.shape[0], img.shape[1])
    cv2.imshow('bilinear interp', imgNew1)
    cv2.waitKey()