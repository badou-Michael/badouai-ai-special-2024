import cv2
import numpy as np
import random

lenna = cv2.imread("./lenna.png")
w, h = lenna.shape[ : 2]
percentage = 0.8
noiseLenna = lenna.copy()
noiseNumber = int(percentage * w * h)
for i in range(noiseNumber):
    randX = random.randint(0, w - 1)
    randY = random.randint(0, h - 1)
    noiseLenna[randX, randY] = lenna[randX, randY] + random.gauss(0, 50)
    for channel in range(3):
        if noiseLenna[randX, randY][channel] < 0:
            noiseLenna[randX, randY][channel] = 0
        elif noiseLenna[randX, randY][channel] > 255:
            noiseLenna[randX, randY][channel] = 255
cv2.imwrite("./gaussianNoise.png", noiseLenna)
cv2.imshow("gaussian noise", noiseLenna)
cv2.waitKey(0)
cv2.destroyAllWindows()