import cv2
import numpy as np
import random

def saltAndPepperNoise(src, percentage):
    noiseImg = src
    noiseNumber = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noiseNumber):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.random() <= 0.5:           
            noiseImg[randX, randY] = 0       
        else:            
            noiseImg[randX, randY] = 255
    return noiseImg

def main():
    lenna = cv2.imread("./lenna.png")
    noiseLenna = saltAndPepperNoise(lenna, 0.1)
    cv2.imwrite("./noiseLenna.png", noiseLenna)
    cv2.imshow("noiseLenna", noiseLenna)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()