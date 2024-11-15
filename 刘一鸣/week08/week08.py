'''
第八周作业：实现两种hash算法。
'''

import cv2
from PIL import Image
import imagehash

# 加载图像
img1 = Image.open('iphone1.png')
img2 = Image.open('iphone2.png')

# 计算均值哈希 (aHash)
hash1 = imagehash.average_hash(img1)
hash2 = imagehash.average_hash(img2)
print(f"均值哈希 (aHash) - Hash1: {hash1}, Hash2: {hash2}")
print(f"均值哈希相似度: {hash1 - hash2}")

# 计算差值哈希 (dHash)
hash1 = imagehash.dhash(img1)
hash2 = imagehash.dhash(img2)
print(f"差值哈希 (dHash) - Hash1: {hash1}, Hash2: {hash2}")
print(f"差值哈希相似度: {hash1 - hash2}")
