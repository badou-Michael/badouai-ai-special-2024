
import cv2
import random

def PepperNoise(src,percetage):
    image = src
    imageNum = int(image.shape[0]*image.shape[1]*percetage)
    for i in range(imageNum):
        x = random.randint(0, image.shape[0]-1)
        y = random.randint(0, image.shape[1]-1)
        if random.random() <= 0.5:
            image[x, y] = 0
        else:
            image[x, y] = 255
    return image


img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg",0)
img1 = PepperNoise(img, 0.1)
img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("source",img2)
cv2.imshow("lenna_GaussianNoise",img1)
cv2.waitKey(0)





