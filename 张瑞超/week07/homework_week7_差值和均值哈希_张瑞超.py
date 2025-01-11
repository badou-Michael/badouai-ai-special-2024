import cv2
import numpy as np


# 计算哈希值通用函数
def compute_hash(img, mode='aHash'):
    if mode == 'aHash':
        # 均值哈希: 缩放为8*8
        img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    elif mode == 'dHash':
        # 差值哈希: 缩放为8*9
        img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("Invalid mode! Use 'aHash' or 'dHash'.")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hash_str = ''

    if mode == 'aHash':
        # 均值哈希算法
        avg = np.mean(gray)  # 使用numpy直接计算平均值
        hash_str = ''.join('1' if gray[i, j] > avg else '0' for i in range(8) for j in range(8))

    elif mode == 'dHash':
        # 差值哈希算法
        hash_str = ''.join('1' if gray[i, j] > gray[i, j + 1] else '0' for i in range(8) for j in range(8))

    return hash_str


# 比较哈希值相似度
def cmp_hash(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1  # 长度不等表示错误
    return sum(h1 != h2 for h1, h2 in zip(hash1, hash2))  # 使用sum和zip简化对比

# 添加高斯噪声函数
def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25  # 噪声的标准差（数值越大，噪声越明显）
    gauss = np.random.normal(mean, sigma, (row, col, ch))  # 生成高斯噪声
    noisy = np.clip(image + gauss, 0, 255).astype(np.uint8)  # 将噪声添加到图像并限制像素范围
    return noisy



# 读取图像并计算哈希值
img1 = cv2.imread('lenna.png')

# 为图像添加噪声
noisy_img = add_noise(img1)

# 保存带有噪声的图像
cv2.imwrite('lenna_noise.png', noisy_img)
img2 = cv2.imread('lenna_noise.png')

# 均值哈希
hash1_aHash = compute_hash(img1, mode='aHash')
hash2_aHash = compute_hash(img2, mode='aHash')
print(f"aHash1: {hash1_aHash}\naHash2: {hash2_aHash}")
similarity_aHash = cmp_hash(hash1_aHash, hash2_aHash)
print(f'均值哈希算法相似度：{similarity_aHash}')

# 差值哈希
hash1_dHash = compute_hash(img1, mode='dHash')
hash2_dHash = compute_hash(img2, mode='dHash')
print(f"dHash1: {hash1_dHash}\ndHash2: {hash2_dHash}")
similarity_dHash = cmp_hash(hash1_dHash, hash2_dHash)
print(f'差值哈希算法相似度：{similarity_dHash}')
