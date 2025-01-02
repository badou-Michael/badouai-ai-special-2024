import random

import cv2

# 高斯噪声
def gaussian_noise(image, mu, sigma, percentage):
    noiseImg = image
    h, w = image.shape[0], image.shape[1]
    noiseNum = int(h * w * percentage)

    for i in range(noiseNum):
        randX = random.randint(0, h - 1)
        randY = random.randint(0, w - 1)
        noiseImg[randX, randY] = int(image[randX, randY] + random.gauss(mu, sigma))
        if noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
        elif noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
    return noiseImg


if __name__ == "__main__":
    lenna = cv2.imread("../images/lenna.png", 0)
    noise = gaussian_noise(lenna, 2, 4, 0.8)
    source = cv2.imread("../images/lenna.png")
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    cv2.imshow("noise", noise)
    cv2.waitKey()
