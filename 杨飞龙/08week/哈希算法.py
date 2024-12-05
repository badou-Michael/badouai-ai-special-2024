import cv2
import numpy as np

# 均值哈希函数
def average_hash(image_path):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, 0)
    # 缩放到8x8
    image = cv2.resize(image, (8, 8),interpolation=cv2.INTER_CUBIC)
    # 计算平均值
    avg = np.mean(image)
    # 生成哈希值，遍历图像的每行（for row in image），然后遍历行中的每个像素值（for pixel in row）

    hash_value = ''.join('1' if pixel > avg else '0' for row in image for pixel in row)
    return hash_value

# 差值哈希函数
def difference_hash(image_path):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, 0)
    # 缩放到8x9,这里的参数传入顺序是(列，行)的顺序
    image = cv2.resize(image, (9, 8),interpolation=cv2.INTER_CUBIC)
    # 计算差分,第二列到最后一列和第一列到倒数第二列 比较大小
    diff = image[:, 1:] < image[:, :-1]
    # 生成哈希值
    hash_value = ''.join('1' if pixel else '0' for row in diff for pixel in row)
    return hash_value

#Hash值对比
def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n


# 测试图像

hash1 = difference_hash("lenna.png")
hash2 = difference_hash("lenna_noise.png")
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('差值哈希算法相似度：', n)

# 测试图像


hash1 = average_hash("lenna.png")
hash2 = average_hash("lenna_noise.png")
print(hash1)
print(hash2)
n = cmpHash(hash1, hash2)
print('均值哈希算法相似度：', n)





