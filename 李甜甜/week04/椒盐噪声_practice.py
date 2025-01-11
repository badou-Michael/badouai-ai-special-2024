import random
import cv2
def fun1(img,percentage):
    Noise = img
    NoiseNum = int(percentage*Noise.shape[0]*Noise.shape[1])
    for i in range(NoiseNum):
        randY = random.randint(0,Noise.shape[0]-1)
        randX = random.randint(0,Noise.shape[1]-1)
        if random.random()>=0.5:
            Noise[randY, randX] = 255
        else:
            Noise[randY, randX] = 0
    return Noise

img = cv2.imread("lenna.png",0)
img2 =fun1(img,0.8)
cv2.imshow("papperand solat",img2)
cv2.waitKey()
