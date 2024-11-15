from imagehash import average_hash, dhash
from PIL import Image
import numpy as np
import cv2

def myMeanHash1(image, hash_size):
    image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    my_mean_value = np.mean(img_gray)
    hash_str = (img_gray > my_mean_value).astype(int)
    return hash_str

def myMeanHash2(image, hash_size):
    image = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    my_mean_value = np.mean(img_gray)
    hash_str = ''.join('1' if i > my_mean_value else '0' for i in img_gray.flatten())
    return hash_str

def mydHash1(image, hash_size):
    image = cv2.resize(image, (hash_size+1, hash_size), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hash_str = np.zeros((hash_size, hash_size))
    for i in range(hash_size):
        for j in range(hash_size):
            if img_gray[i,j] > img_gray[i, j+1]:
                hash_str[i,j] = 1
            else:
                hash_str[i,j] = 0
    return hash_str

def mydHash2(image, hash_size):
    image = cv2.resize(image, (hash_size+1, hash_size), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hash_str = ''
    for i in range(hash_size):
        for j in range(hash_size):
            if img_gray[i,j] > img_gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

def cmpHash(image1, image2):
    return np.sum(image1 != image2)

def cmpHashHamming(str1, str2):
    distance = 0
    if len(str1)!=len(str2):
        return -1

    for char1, char2 in zip(str1, str2):
        if char1 != char2:
            distance += 1
    return distance

#导入图片数据
image = cv2.imread("lenna.png")
# 生成和图像大小一样的随机噪声
random_noise = np.random.randint(0, 256, image.shape, dtype=np.uint8)
# 将随机噪声添加到图像中
noisy_image = cv2.add(image, random_noise)
cv2.imshow('Original', image)
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)

#计算哈希值--均值哈希
hash1 = myMeanHash1(image, 8)
print(hash1)
hash1_noise = myMeanHash1(noisy_image, 8)
print(hash1_noise)
diff_hash1 = cmpHash(hash1, hash1_noise)
print(diff_hash1)

hash2 = myMeanHash2(image, 8)
print(hash2)
hash2_noise = myMeanHash2(noisy_image, 8)
print(hash2_noise)
diff_hash2 = cmpHashHamming(hash2, hash2_noise)
print(diff_hash2)

#计算哈希值--差分哈希
hash3 = mydHash1(image, 8)
print(hash3)
hash3_noise = mydHash1(noisy_image, 8)
print(hash3_noise)
diff_hash3 = cmpHash(hash3, hash3_noise)
print(diff_hash3)

hash4 = mydHash2(image, 8)
print(hash4)
hash4_noise = mydHash2(noisy_image, 8)
print(hash4_noise)
diff_hash4 = cmpHashHamming(hash4, hash4_noise)
print(diff_hash4)


##使用已有的库函数实现均值哈希和差分哈希
image_hash1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hash1 = Image.fromarray(image_hash1)

image_noise_hash1 = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
image_noise_hash1 = Image.fromarray(image_noise_hash1)

#均值哈希
hash_value1 = average_hash(image_hash1)
hash_value1_noise = average_hash(image_noise_hash1)
print(hash_value1)
print(hash_value1_noise)
# 计算两个哈希值之间的汉明距离
print(hash_value1 - hash_value1_noise)

#差分哈希
hash_value2 = dhash(image_hash1)
hash_value2_noise = dhash(image_noise_hash1)
print(hash_value2)
print((hash_value2_noise))
print(hash_value2 - hash_value2_noise)
