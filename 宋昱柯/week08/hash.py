import cv2 as cv
import numpy as np


def a_hash(img, hash_size=8):
    """均值哈希"""
    # 图像预处理
    img = cv.cvtColor(
        cv.resize(img, (hash_size, hash_size), interpolation=cv.INTER_CUBIC),
        cv.COLOR_BGR2GRAY,
    )

    # 返回哈希值
    return "".join(["1" if p > img.mean() else "0" for p in img.ravel()])


def d_hash(img, hash_size=8):
    """差值哈希"""
    # 图像预处理
    img = cv.cvtColor(
        cv.resize(img, (hash_size, hash_size + 1), interpolation=cv.INTER_CUBIC),
        cv.COLOR_BGR2GRAY,
    )
    # 计算差值
    diffrence = img[:, :-1] > img[:, 1:]
    # 返回哈希值
    return "".join(["1" if d else "0" for d in diffrence.ravel()])


def cmphash(hash_a: str, hash_b: str):
    """计算汉明距离"""
    if not (isinstance(hash_a, str) and isinstance(hash_b, str)):
        raise TypeError("哈希值必须是字符串类型")

    if len(hash_a) != len(hash_b):
        raise ValueError("哈希值长度必须相同")

    distance = np.sum(ch_a != ch_b for ch_a, ch_b in zip(hash_a, hash_b))

    return distance

img1 = cv.imread("practice\cv\week08\lenna.png")
img2 = img1.copy()
img2 = cv.GaussianBlur(img2,(5,5),1)

hash1 = a_hash(img1)
hash2 = a_hash(img2)
print(hash1)
print(hash2)
n = cmphash(hash1, hash2)
print("均值哈希算法相似度：", n)

hash1 = d_hash(img1)
hash2 = d_hash(img2)
print(hash1)
print(hash2)
n = cmphash(hash1, hash2)
print("差值哈希算法相似度：", n)
