import cv2
import numpy as np

imread = cv2.imread("../sea.jpg")

cv2.resize(imread, (8, 8), )


def make_noise(image, mean=0, sigma=15):
    """添加高斯噪声到图像"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian_noise, 0, 255)
    return noisy_image.astype(np.uint8)


# 均值hash
def hash_avg(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s += gray[i, j]
    avg = s / 64
    # 大于平均值为1，反之为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值hash
def hash_diff(img):
    # 缩放为8*9
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个大于后一个就为1，反之为0
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def hash_compare(hash1, hash2):
    diff_count = 0
    hash_len = len(hash1)
    if len(hash2) != hash_len:
        return -1
    for i in range(hash_len):
        if hash1[i] != hash2[i]:
            diff_count += 1
    return diff_count


img1 = cv2.imread('../sea.jpg')
img2 = make_noise(img1)

hash1 = hash_avg(img1)
hash2 = hash_avg(img2)
print(hash1)
print(hash2)
n = hash_compare(hash1, hash2)
print('均值hash算法图像相似度: ', n)

hash1 = hash_diff(img1)
hash2 = hash_diff(img2)
print(hash1)
print(hash2)
n = hash_compare(hash1, hash2)
print('差值hash算法图像相似度: ', n)
