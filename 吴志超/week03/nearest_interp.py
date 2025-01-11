import cv2
import numpy as np

def solution(img):
    height,width,channels = img.shape
    emptyImg = np.zeros((1000,1000,channels),np.uint8)
    sh = 1000/height
    sw = 1000/width
    for i in range(1000):
        for j in range(1000):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImg[i,j] = img[x,y]

    return emptyImg


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    testImg = cv2.resize(img,(1000,1000),cv2.INTER_NEAREST)
    newImg = solution(img)
    cv2.imshow("newImage",newImg)
    cv2.imshow("origin",img)
    cv2.imshow("test",testImg)
    cv2.waitKey(0)
