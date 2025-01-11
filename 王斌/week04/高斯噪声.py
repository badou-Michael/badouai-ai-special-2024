
import cv2
import random

def GaussianNoise(src,means,sigma,percetage):
    image = src
    imageNum = int(image.shape[0]*image.shape[1]*percetage*image.shape[2])
    for i in range(imageNum):
        x = random.randint(0, image.shape[0]-1)
        y = random.randint(0, image.shape[1]-1)
        z = random.randint(0, image.shape[2]-1)
        image[x, y, z] = image[x, y, z]+random.gauss(means, sigma)
        if image[x, y, z]< 0:
            image[x, y, z] = 0
        elif image[x, y, z] > 255:
            image[x, y, z] = 255
    return image




img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
img1 = GaussianNoise(img, 2, 4, 0.8)
img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("source",img)
cv2.imshow("lenna_GaussianNoise",img1)
cv2.waitKey(0)





