
import cv2
import random


def gauss_noise(img_source, mean_value, sigma_value, percentage):
    img_noise = img_source
    num_noise = int(percentage * img_noise.shape[0] * img_noise.shape[1])
    for i in range(num_noise):
        # 每次取一个随机点
        x_rand = random.randint(0, img_noise.shape[0] -1)
        y_rand = random.randint(0, img_noise.shape[1] -1)

        # 在原有像素灰度值上 加上高斯随机数
        # mu（均值）：这是高斯分布的中心值，也就是分布的平均水平。在正态分布曲线中，它是最高点所在的横坐标。
        # sigma（标准差）：这是高斯分布的宽度，表示数据分布的离散程度。标准差越大，数据分布越分散；标准差越小，数据分布越集中。在正态分布曲线中，它控制了曲线的“胖瘦”。
        img_noise[x_rand, y_rand] = img_noise[x_rand, y_rand] + random.gauss(mean_value, sigma_value)

        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if img_noise[x_rand, y_rand] < 0:
            img_noise[x_rand, y_rand] = 0
        elif img_noise[x_rand, y_rand] > 255:
            img_noise[x_rand, y_rand] = 255

    return img_noise

img_src = cv2.imread('lenna.png', 0)  # flag=0 灰色模式
cv2.imshow('source gray', img_src)

img_gauss = gauss_noise(img_src, 2, 8, 0.8)
cv2.imshow('after gauss', img_gauss)

cv2.waitKey(0)
cv2.destroyAllWindows()