import random

import cv2

# 椒盐噪声
def jyNoise(img, percentage):
    noiseImg = img
    h, w = img.shape[0], img.shape[1]
    noiseNum = int(h * w * percentage)
    for i in range(noiseNum):
        randX = random.randint(0, h - 1)
        randY = random.randint(0, w - 1)
        if random.random() >= 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg


if __name__ == "__main__":
    img = cv2.imread("../images/lenna.png", 0)
    noise = jyNoise(img, 0.5)
    img1 = cv2.imread("../images/lenna.png")
    source = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # print(img - source)
    cv2.imshow("noise", noise)
    cv2.imshow("lenna", img)
    cv2.waitKey(0)
