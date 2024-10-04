import random
import cv2


def pepper_salt_noise(img, percentage):
    NoiseImg = img
    # 总的噪声点
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        # 图片边缘不做处理，故-1，随机取行、列
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        # 随机生成椒盐噪声
        if random.random() < 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
# 噪声的占比比例
percentage = 0.8
img1 = pepper_salt_noise(img, percentage)
cv2.imshow('pepper_salt_noise', img1)
cv2.waitKey(0)
cv2.distroyAllWindows()
