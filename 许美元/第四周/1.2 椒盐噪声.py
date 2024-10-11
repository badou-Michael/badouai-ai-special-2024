
import cv2
import random


def pepper_salt_noise(img_source, percentage):
    img_noise = img_source
    num_noise = int(percentage * img_noise.shape[0] * img_noise.shape[1])
    for i in range(num_noise):
        # 每次取一个随机点
        x_rand = random.randint(0, img_noise.shape[0] -1)
        y_rand = random.randint(0, img_noise.shape[1] -1)

        #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            img_noise[x_rand, y_rand] = 0
        else:
            img_noise[x_rand, y_rand] = 255

    return img_noise

img_src = cv2.imread('lenna.png', 0)  # flag=0 灰色模式
cv2.imshow('source gray', img_src)

img_ps = pepper_salt_noise(img_src, 0.8)
cv2.imwrite('lenna_PepperandSalt.png', img_ps)
cv2.imshow('after pepper&salt', img_ps)

cv2.waitKey(0)
cv2.destroyAllWindows()