import cv2


def get_hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1
    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


# 1.均值哈希算法
def get_avg_hash(image):
    # 缩放图像到8*8
    resize_image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 转灰度图
    resize_image_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    # 求全部像素平均值
    mean_value = cv2.mean(resize_image_gray)
    avg = mean_value[3]
    # 像素值和平均值进行比较,生成图片哈希
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if resize_image_gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def avg_hash():
    img_hash = get_avg_hash(img)
    img_noise_hash = get_avg_hash(img_noise)
    # 对比汉明距离
    hamming_distance = get_hamming_distance(img_hash, img_noise_hash)
    print('avg_hash: ', hamming_distance)


# 2.插值哈希算法
def get_dif_hash(image):
    # 缩放图像到8*9
    resize_image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转灰度图
    resize_image_gray = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
    # 像素值和同行后一位比较
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if resize_image_gray[i, j] > resize_image_gray[i, j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dif_hash():
    img_hash = get_dif_hash(img)
    img_noise_hash = get_dif_hash(img_noise)
    # 对比汉明距离
    hamming_distance = get_hamming_distance(img_hash, img_noise_hash)
    print('dif_hash: ', hamming_distance)


img = cv2.imread('lenna_GaussianNoise.png')
img_noise = cv2.imread('lenna_spNoise.png')
avg_hash()
dif_hash()
