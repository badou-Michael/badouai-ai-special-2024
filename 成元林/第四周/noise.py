import cv2 as cv
import numpy as np
from numba import njit
import random
def add_gaussian_noise(image, mean=0, std=25,amount=0.05):
    """

    @param image: 原图像
    @param mean: 均值
    @param std: 标准差
    @param amount: 噪声数量占比
    @return:
    """
    noise_image = np.copy(image)
    height,wihdth = noise_image.shape[:2]
    # 求得噪声数量
    noise_mount = int(amount*height*wihdth)
    for i in range(noise_mount):
        # 生成随机添加噪声的像素点
        rand_x = random.randint(0,height-1)
        rand_y = random.randint(0,wihdth-1)
        noise_image[rand_x,rand_y] = noise_image[rand_x,rand_y]+random.gauss(mean,std)
        # 大于255的值取255，小于0的值为0
        noise_image = np.clip(noise_image,0,255)
    return noise_image

@njit
def add_salt_peper_noise(image,s_vs_p=0.5,amount=0.005):
    """
    添加椒盐噪声
    @param image: 原始图像
    @param s_vs_p: 椒盐噪声比例（椒和盐的比例）
    @param amount: 噪声的量
    @return: 添加噪声后的图像
    """
    noise_image = image.copy()
    # 获取图像的高和宽
    height,width = noise_image.shape[:2]
    #计算噪声的量
    noise_mount = amount*height*width
    #椒的数量
    p_num = noise_mount*(1-s_vs_p)
    # 盐的数量
    s_num = noise_mount*s_vs_p
    for i in range(p_num):
        rand_x = random.randint(0,height-1)
        rand_y = random.randint(0,width-1)
        noise_image[rand_x,rand_y] = 0
    for j in range(s_num):
        rand_x1 = random.randint(0,height-1)
        rand_y1 = random.randint(0,width-1)
        noise_image[rand_x1,rand_y1]=255
    return noise_image

if __name__ == '__main__':
    ori_image = cv.imread("lenna.png")
    # noise_image = add_gaussian_noise(ori_image)
    noise_image = add_salt_peper_noise(ori_image)
    cv.imshow("ori_image",ori_image)
    cv.imshow("noise_image",noise_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # salt_peper = np.ones((10, 10), dtype=bool)
    # salt_peper[:4] = 0
    # print(salt_peper)
    # np.random.shuffle(salt_peper)
    # print(salt_peper)