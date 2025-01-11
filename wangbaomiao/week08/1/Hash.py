# -*- coding: utf-8 -*-
# time: 2024/11/13 17:13
# file: Hash.py
# author: flame
import cv2

"""
定义两个哈希函数aHash和pHash，分别计算图像的平均哈希值和感知哈希值。
同时定义一个cmpHash函数，用于比较两个哈希值的相似度。
最后，读取两张图像，计算它们的哈希值并输出相似度。
"""

""" 导入OpenCV库，简称为cv2，用于图像处理。 """
import cv2

""" 定义平均哈希函数aHash，计算图像的平均哈希值。 """
def aHash(img):
    """
    计算图像的平均哈希值。

    参数:
    img: 输入的图像。

    返回:
    hash_str: 生成的哈希字符串。
    """
    """ 缩放图像至8x8像素，使用立方插值法以保持图像质量。 """
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)

    """ 将图像转换为灰度图，减少颜色信息的影响。 """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """ 初始化灰度值总和s和哈希字符串hash_str。 """
    s = 0
    hash_str = ''

    """ 遍历8x8的灰度图，计算所有像素的灰度值总和。 """
    for i in range(8):
        for j in range(8):
            ''' gray 是一个二维数组，表示8x8的灰度图像。
            gray[i, j] 表示第 i 行第 j 列的像素值。这个值是一个0到255之间的整数，表示该像素的灰度值。'''
            s = s + gray[i, j]

    """ 计算灰度值的平均值。 """
    avg = s / 64

    """ 遍历8x8的灰度图，根据每个像素的灰度值与平均值的比较结果生成哈希字符串。 """
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    """ 返回生成的哈希字符串。 """
    return hash_str

""" 定义感知哈希函数pHash，计算图像的感知哈希值。 """
def pHash(img):
    """
    计算图像的感知哈希值。

    参数:
    img: 输入的图像。

    返回:
    hash_str: 生成的哈希字符串。
    """
    """ 缩放图像至8x9像素，使用立方插值法以保持图像质量。 """
    img = cv2.resize(img, (8, 9), interpolation=cv2.INTER_CUBIC)

    """ 将图像转换为灰度图，减少颜色信息的影响。 """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """ 初始化哈希字符串hash_str。 """
    hash_str = ''

    """ 遍历8x8的灰度图，根据每个像素与其下方像素的灰度值比较结果生成哈希字符串。 """
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i + 1, j]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    """ 返回生成的哈希字符串。 """
    return hash_str

""" 定义比较哈希值相似度的函数cmpHash。 """
def cmpHash(hash1, hash2):
    """
    比较两个哈希值的相似度。

    参数:
    hash1: 第一个哈希字符串。
    hash2: 第二个哈希字符串。

    返回:
    n: 两个哈希字符串的不同位数。
    """
    """ 检查两个哈希字符串的长度是否相同，如果不同则返回-1表示无法比较。 """
    if len(hash1) != len(hash2):
        return -1

    """ 初始化不同位数计数器n。 """
    n = 0

    """ 遍历两个哈希字符串，统计不同位数的数量。 """
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1

    """ 返回不同位数的数量。 """
    return n

""" 读取第一张图像'lenna.png'，并存储在变量img1中。 """
img1 = cv2.imread("lenna.png")

""" 读取第二张图像'lenna_noise.png'，并存储在变量img2中。 """
img2 = cv2.imread("lenna_noise.png")

""" 计算第一张图像的平均哈希值，并存储在变量hash1中。 """
hash1 = aHash(img1)

""" 计算第二张图像的平均哈希值，并存储在变量hash2中。 """
hash2 = aHash(img2)

""" 输出两张图像的平均哈希值。 """
print("ahash1 -> : ", hash1, "\nahash2 -> : ", hash2)

""" 计算两张图像平均哈希值的相似度，并存储在变量n中。 """
n = cmpHash(hash1, hash2)

""" 输出两张图像平均哈希值的相似度。 """
print("均值哈希相似度 -> : ", n)

""" 计算第一张图像的感知哈希值，并存储在变量hash1中。 """
hash1 = pHash(img1)

""" 计算第二张图像的感知哈希值，并存储在变量hash2中。 """
hash2 = pHash(img2)

""" 输出两张图像的感知哈希值。 """
print("phash1 -> : ", hash1, "\nphash2 -> : ", hash2)

""" 计算两张图像感知哈希值的相似度，并存储在变量n中。 """
n = cmpHash(hash1, hash2)

""" 输出两张图像感知哈希值的相似度。 """
print("感知哈希相似度 -> : ", n)
