import cv2
import numpy as np
import random
def pepperSaltNoise(src, SNR):
    pepper_picture = np.copy(src)
    size = int(pepper_picture.shape[0] * pepper_picture.shape[1] * SNR)
    for i in range(size):
        random_x = random.randint(0, pepper_picture.shape[0] - 1)
        random_y = random.randint(0, pepper_picture.shape[1] - 1)
        possibility = random.random()
        if possibility < 0.5:
            pepper_picture[random_x][random_y] = 0
        else:
            pepper_picture[random_x][random_y] = 255
    return pepper_picture

src = cv2.imread("lenna.png", 0)
pepper_picture = pepperSaltNoise(src, 0.005)
src1 = cv2.imread("lenna.png")
src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
cv2.imshow("original", src1)
cv2.imshow("pepper", pepper_picture)
cv2.waitKey()
