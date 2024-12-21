import cv2
import numpy as np

# 均值哈希算法
def average_hash(img, size=8):
    # 缩放为 8 x 8
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # 计算灰度平均值
    avg = gray.mean()
    # 生成哈希值
    hash_str = ''.join('1' if pixel > avg else '0' for row in gray for pixel in row)
    return hash_str

# 差值哈希算法
def difference_hash(img, size=8):
    # 缩放为 (size+1) x size
    resized = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # 生成哈希值
    hash_str = ''.join(
        '1' if gray[row, col] > gray[row, col + 1] else '0'
        for row in range(size) for col in range(size)
    )
    return hash_str

# 哈希值对比
def hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1
    # 返回汉明距离
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def main():
    img1 = cv2.imread('lenna.jpg')
    img2 = cv2.imread('lenna_noise.jpg')

    # 检查图片是否加载成功
    if img1 is None or img2 is None:
        print("Error: One or both images could not be loaded. Please check the file paths.")
        return

    # 均值哈希算法
    hash1 = average_hash(img1)
    hash2 = average_hash(img2)
    print("均值哈希值1:", hash1)
    print("均值哈希值2:", hash2)
    print("均值哈希算法相似度:", hamming_distance(hash1, hash2))

    # 差值哈希算法
    hash1 = difference_hash(img1)
    hash2 = difference_hash(img2)
    print("差值哈希值1:", hash1)
    print("差值哈希值2:", hash2)
    print("差值哈希算法相似度:", hamming_distance(hash1, hash2))

if __name__ == '__main__':
    main()
