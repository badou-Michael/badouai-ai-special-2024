
import cv2
import random

def jiaoyanNoice(src, percentage):
    img = src
    num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(num):
        srcX = random.randint(0, src.shape[0] - 1)
        srcY = random.randint(0, src.shape[1] - 1)

        if random.random() <= 0.5:
            img[srcX, srcY] = 0
        else:
            img[srcX, srcY] = 255
    return img

img = cv2.imread("test_img.jpeg", 0)
img1 = jiaoyanNoice(img,  0.01)
img = cv2.imread("test_img.jpeg")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("source", img2)
cv2.imshow("gosi_noice", img1)
cv2.waitKey(0)