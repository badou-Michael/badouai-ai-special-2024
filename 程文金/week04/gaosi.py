
import cv2
import random

def gaoSiNoice(src, means, sigma, percentage):
    img = src
    num = int( img.shape[0] * img.shape[1] * percentage )
    for i in range(num):
        srcX = random.randint(0, img.shape[0] - 1)
        srcY = random.randint(0, img.shape[1] - 1)

        value = img[srcX, srcY] + random.gauss(means, sigma)
        img[srcX, srcY] = value
        if img[srcX, srcY] < 0:
            img[srcX, srcY] = 0
        elif img[srcX, srcY] > 255:
            img[srcX, srcY] = 255
    return img


img = cv2.imread("test_img.jpeg", 0)
img1 = gaoSiNoice(img, 3, 5, 0.7)
img = cv2.imread("test_img.jpeg")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("source", img2)
cv2.imshow("gosi_noice", img1)
cv2.waitKey(0)