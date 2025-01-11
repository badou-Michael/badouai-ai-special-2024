# 高斯实现方式总结

import cv2
import matplotlib.pyplot as plt
from skimage import util

import numpy as np
import random

# 法一：random.gauss取gauss值，手动限制范围
def add_gaussian_noise_mth1(image, mean=0, sigma=25, noise_ratio=1):
    noisy_image = image.copy() # 深拷贝！！！

    if False:
        noisy_num = int(noise_ratio * image.shape[0] * image.shape[1])
        for i in range(noisy_num):
            randY = int(random.randint(0, image.shape[0] - 1))
            randX = int(random.randint(0, image.shape[1] - 1))
            if True:
                random_gauss = random.gauss(mean, sigma)
                noisy_image[randY, randX] = noisy_image[randY, randX] + random_gauss

                if noisy_image[randY, randX] < 0:
                    noisy_image[randY, randX] = 0
                elif noisy_image[randY, randX] > 255:
                    noisy_image[randY, randX] = 255
            else:
                noisy_image[randY, randX] = np.clip(noisy_image[randY, randX] + random.gauss(mean, sigma), 0, 255)
    else:
        noisy_num = noise_ratio * image.shape[0] * image.shape[1]
        noise = [random.gauss(mean, sigma) for i in range(noisy_num)]

        for i in range(noisy_num):
            randY = int(random.randint(0, image.shape[0] - 1))
            randX = int(random.randint(0, image.shape[1] - 1))
            noisy_image[randY, randX] = np.clip((noisy_image[randY, randX] + noise[i]), 0, 255)
    return noisy_image


# 法二：np.random.normal取gauss值，np.clip限制范围:https://blog.csdn.net/sinat_29957455/article/details/123977298
def add_gaussian_noise_mth2(image, mean=0, sigma=25):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma**0.5, (row, col))  #normal的参数是标准差
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image  # TypeError: Image data of dtype object cannot be converted to float


# 法三：cv2.randn()生成噪声，cv2.add添加到图像， np.clip限制范围
def add_gaussian_noise_mth3(image, mean=0, sigma=25):
    print(image, image.shape, type(image), image.dtype)
    noise = np.zeros_like(image, dtype=np.float32)  # 大小一样的0数组
    print(noise, noise.shape, type(noise), noise.dtype)
    cv2.randn(noise, mean, sigma)  # 生成噪声
    if noise.dtype != image.dtype:
        image = image.astype(noise.dtype)  # 数据类型转成一样的
    noisy_image = cv2.add(image, noise)  # 添加噪声  # TypeError:  error: (-5:Bad argument) When the input arrays in add/subtract/multiply/divide functions have different types,
    noisy_image = np.clip(noisy_image, 0, 255)  # 限制范围
    return noisy_image.astype(np.uint8)  # 转换类型


# 法四：调接口：使用skimage的util.random_noise，指定mode
def add_gaussian_noise_mth4(image, mean=0, sigma=25):
    noisy_image = util.random_noise(image, mode='gaussian', var=sigma)
    return noisy_image


# ---其他---：
# 法五，scipy.stats.norm()生成高斯噪声值，
# 法六，TensorFlow:tf.random.normal生成高斯噪声（tf读取图像）
# torch.randn生成高斯值
# 实现高斯模糊而非添加高斯噪声：cv2.GaussianBlur https://pythonjishu.com/bdzvlxeuwaqnnbn/
# https://blog.csdn.net/howlclat/article/details/107216722


if __name__ == "__main__":
    orig_img = cv2.imread("lenna.png", flags=0)
    plt.subplot(241), plt.imshow(orig_img, cmap='gray'), plt.title("orig_img"), plt.axis('off')

    img_method_one = add_gaussian_noise_mth1(orig_img, mean=0, sigma=25)
    plt.subplot(242), plt.imshow(img_method_one, cmap='gray'), plt.title("img_method_one"), plt.axis('off')

    img_method_two = add_gaussian_noise_mth2(orig_img, mean=0, sigma=25)
    plt.subplot(243), plt.imshow(img_method_two, cmap='gray'), plt.title("img_method_two"), plt.axis('off')

    img_method_three = add_gaussian_noise_mth3(orig_img, mean=0, sigma=25)
    plt.subplot(244), plt.imshow(img_method_three, cmap='gray'), plt.title("img_method_three"), plt.axis('off')
    
    img_method_four = add_gaussian_noise_mth4(orig_img, mean=0, sigma=0.01)
    plt.subplot(245), plt.imshow(img_method_four,  cmap='gray'), plt.title("img_method_four"), plt.axis('off')

    plt.show()
