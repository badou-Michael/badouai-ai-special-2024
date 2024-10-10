import random
import matplotlib.pyplot as plt
from skimage import util
import cv2
import numpy as np

# 法一：random.random在0-1内抛硬币，决定是椒还是盐
def pepper_salt_noise_mth1(image, noise_ratio=0.1):
    noisy_img = image.copy()
    noise_num = int(noise_ratio * image.shape[0] * image.shape[1])  # 或者int(noise_ratio * image.size)

    # print(image.shape)
    # print(image.size)

    # 取随机像素点进行赋值
    for i in range(noise_num):
        # random.randint生成随机整数
        randY = random.randint(0, noisy_img.shape[0] - 1)
        randX = random.randint(0, noisy_img.shape[1] - 1)

        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5:
            noisy_img[randY, randX] = 0
        else:
            noisy_img[randY, randX] = 255

    return noisy_img

# 法二：处理两遍噪点，第一遍给椒，第二遍加盐
def pepper_salt_noise_mth2(image, noise_level=0.1):  # noise_level噪声级，默认2%的噪声
    noisy_image = image.copy()
    num_noise_pixels = int(noise_level * image.size)  # 噪声像素数

    coords = [np.random.randint(0, i - 1, num_noise_pixels // 2) for i in image.shape]  # //整除
    noisy_image[coords[0], coords[1]] = 255  # 替换为白色
    # print(image.shape)
    # for i in image.shape:
    #     print(i)
    #     break
    # print(coords)

    coords = [np.random.randint(0, i - 1, num_noise_pixels // 2) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # 替换为黑色

    return noisy_image


# 法三：添加噪声层
def pepper_salt_noise_mth3(image, noise_level=0.1):
    noisy_image = image.copy()

    # 一个与输入图像 image 相同形状的随机矩阵，矩阵中的值在 0 到 1 之间均匀分布
    salt_pepper = np.random.rand(*image.shape)
    print(salt_pepper)

    # 因为salt_pepper中的0-1均匀分布，所以按照噪声比，可以保留这部分作为椒盐噪声
    # 盐噪声（白点）
    noisy_image[salt_pepper < (noise_level / 2)] = 255  # 替换为白色
    # 胡椒噪声（黑点）
    noisy_image[salt_pepper > (1 - noise_level / 2)] = 0  # 替换为黑色

    return noisy_image

# 法四：skimage的util.random_noise接口
def pepper_salt_noise_mth4(image, noise_ratio=0.1):
    noisy_img = util.random_noise(image, mode="s&p", amount = noise_ratio)
    return noisy_img

if __name__ == "__main__":
    orig_img = cv2.imread("lenna.png", flags=0)
    plt.subplot(231), plt.imshow(orig_img, cmap='gray'), plt.title("orig_img"), plt.axis('off')

    img_method_one = pepper_salt_noise_mth1(orig_img)
    plt.subplot(232), plt.imshow(img_method_one, cmap='gray'), plt.title("img_method_one"), plt.axis('off')

    img_method_two = pepper_salt_noise_mth2(orig_img)
    plt.subplot(233), plt.imshow(img_method_two, cmap='gray'), plt.title("img_method_two"), plt.axis('off')
    #
    img_method_three = pepper_salt_noise_mth3(orig_img)
    plt.subplot(234), plt.imshow(img_method_three, cmap='gray'), plt.title("img_method_three"), plt.axis('off')

    img_method_four = pepper_salt_noise_mth4(orig_img)
    plt.subplot(235), plt.imshow(img_method_four,  cmap='gray'), plt.title("img_method_four"), plt.axis('off')


    plt.show()
