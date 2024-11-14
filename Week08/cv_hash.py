import cv2
import numpy as np

'''
均值哈希算法

步骤
1. 缩放: 图片缩放为8*8, 保留结构, 除去细节。
2. 灰度化: 转换为灰度图。
3. 求平均值: 计算灰度图所有像素的平均值。
4. 比较: 像素值大于平均值记作1, 相反记作0, 总共64位。
5. 生成hash: 将上述步骤生成的1和0按顺序组合起来既是图片的指纹(hash)。
6. 对比指纹: 将两幅图的指纹对比, 计算汉明距离, 即两个64位的hash值有多少位是不一样的, 不相同位数越少, 图片越相似。
'''
'''
差值哈希算法

差值哈希算法相较于均值哈希算法, 前期和后期基本相同, 只有中间比较hash有变化。

步骤
1. 缩放: 图片缩放为8*9, 保留结构, 除去细节。
2. 灰度化: 转换为灰度图。
3. 求平均值: 计算灰度图所有像素的平均值。 ---这步没有, 只是为了与均值哈希做对比
4. 比较: 像素值大于后一个像素值记作1, 相反记作0。本行不与下一行对比, 每行9个像素, 八个差值, 有8行, 总共64位
5. 生成hash: 将上述步骤生成的1和0按顺序组合起来既是图片的指纹(hash)。
6. 对比指纹: 将两幅图的指纹对比, 计算汉明距离, 即两个64位的hash值有多少位是不一样的, 不相同位数越少, 图片越相似。
'''

# 均值哈希算法 (aHash)
def average_hash(image):
    # 将图像缩放为 8x8 像素
    resized_image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC) # interpolation=cv2.INTER_CUBIC 双三次插值 目的是为了保持图片的清晰度
    # 转换为灰度图
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # 初始化像素值总和和哈希字符串
    pixel_sum = 0
    hash_string = ''
    
    # 计算所有像素值的总和
    for row in range(8):
        for col in range(8):
            pixel_sum += gray_image[row, col]
    
    # 计算平均灰度值
    avg_pixel_value = pixel_sum / 64
    
    # 根据像素值是否大于平均值生成哈希值
    for row in range(8):
        for col in range(8):
            if gray_image[row, col] > avg_pixel_value:
                hash_string += '1'
            else:
                hash_string += '0'
                
    return hash_string

# 差值哈希算法 (dHash)
def difference_hash(image):
    # 将图像缩放为 9x8 像素
    resized_image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 转换为灰度图
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    hash_string = ''
    
    # 比较相邻像素生成哈希值
    for row in range(8):
        for col in range(8):
            if gray_image[row, col] > gray_image[row, col + 1]:
                hash_string += '1'
            else:
                hash_string += '0'
                
    return hash_string

# 对比两个哈希值，计算不同位数
def compare_hash(hash1, hash2):
    # 如果哈希值长度不同，返回 -1 表示出错
    if len(hash1) != len(hash2):
        return -1
    
    # 统计不同位数，表示相似度
    difference_count = sum(1 for i in range(len(hash1)) if hash1[i] != hash2[i])
    return difference_count

# 读取图像
image1 = cv2.imread('lenna.png')
image2 = cv2.imread('lenna_noise.png')

# 计算均值哈希并对比
a_hash1 = average_hash(image1)
a_hash2 = average_hash(image2)
print("图像 1 的均值哈希:", a_hash1)
print("图像 2 的均值哈希:", a_hash2)
a_hash_similarity = compare_hash(a_hash1, a_hash2)
print('均值哈希相似度:', a_hash_similarity)

# 计算差值哈希并对比
d_hash1 = difference_hash(image1)
d_hash2 = difference_hash(image2)
print("图像 1 的差值哈希:", d_hash1)
print("图像 2 的差值哈希:", d_hash2)
d_hash_similarity = compare_hash(d_hash1, d_hash2)
print('差值哈希相似度:', d_hash_similarity)


'''
from PIL import Image
import imagehash # 需要安装 imagehash 库 (pip/conda install imagehash)

# 加载图像
image1 = Image.open('lenna.png')
image2 = Image.open('lenna_noise.png')

# 计算不同类型的哈希值
hash1 = imagehash.average_hash(image1)  # 均值哈希 (aHash)
hash2 = imagehash.average_hash(image2)

print("图像1的均值哈希:", hash1)
print("图像2的均值哈希:", hash2)
print("均值哈希相似度:", hash1 - hash2)  # 使用减法计算哈希之间的差异

# 计算其他哈希类型的示例
hash1_d = imagehash.dhash(image1)  # 差值哈希 (dHash)
hash2_d = imagehash.dhash(image2)
print("差值哈希相似度:", hash1_d - hash2_d)

hash1_p = imagehash.phash(image1)  # 感知哈希 (pHash)
hash2_p = imagehash.phash(image2)
print("感知哈希相似度:", hash1_p - hash2_p)

hash1_w = imagehash.whash(image1)  # 小波哈希 (wHash)
hash2_w = imagehash.whash(image2)
print("小波哈希相似度:", hash1_w - hash2_w)
'''
'''
说明
imagehash 库支持的哈希类型:

aHash: 平均哈希, 适用于简单相似度比较。
dHash: 差值哈希, 对图像内容的细微差异较为敏感。
pHash: 感知哈希, 适用于不同分辨率和格式的图像比较。
wHash: 小波哈希, 可以更准确地捕捉图像细节。
哈希差异: 通过减法运算 (hash1 - hash2) 可以得到两个哈希值的汉明距离, 距离越小, 图像越相似。
'''

