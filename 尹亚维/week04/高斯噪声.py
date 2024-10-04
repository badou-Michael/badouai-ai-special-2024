import random
import cv2


def gaussian_noise(img, means, sigma, percentage):
    NoiseImg = img
    # 总的噪声点
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    # 对每个噪声点进行处理
    for i in range(NoiseNum):
        # 图片边缘不做处理，故-1
        # 随机生成的行
        randX = random.randint(0, img.shape[0] - 1)
        # 随机生成的列
        randY = random.randint(0, img.shape[1] - 1)
        # 在随机生成的位置加上高斯噪声
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
        elif NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
    return NoiseImg


# cv2.imread(filename, flags=None)
# flags:读取模式标志， 默认值为-1（自动检测颜色通道），1：彩色图像， 0：灰度模式
img = cv2.imread('lenna.png', 0)
# 分布的均值，表示分布的中心位置
means = 10
# 分布的标准差
sigma = 100
# 随机取百分之多少比例的像素做高斯噪声
percentage = 0.95
img1 = gaussian_noise(img, means, sigma, percentage)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Origin img", img2)
cv2.imshow("Gaussian Noise", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
