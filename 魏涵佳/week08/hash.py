
import cv2
import numpy as np


def Gaussian_noise(img):
    # 获取图像的尺寸
    height, width, channels = img.shape

    # 生成与图像大小相同的高斯噪声
    mean = 0
    std_dev = 25
    gaussian_noise = np.random.normal(mean, std_dev, (height, width, channels)).astype(np.uint8)

    # 将高斯噪声添加到原始图像
    gaussian_noise = gaussian_noise/(height*width)
    noisy_img = cv2.add(img, gaussian_noise)

    return noisy_img


# 均值hash算法
def aHash(img):
    # 缩放8*8
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    '这里用双三次插值是一种高精度插值方法，适用于需要高质量图像的场景'
    # 转化为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初始为''
    s = 0
    hash_str = ''
    # 遍历求像素值的和
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    # 求平均灰度
    avg = s/64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# 差值hash
def dHash(img):
    # 缩放成8*9
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    "(9，8)指的是宽度是9，高度是8（不是行数，列数）！"
    # 转化为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素值大于后一个像素值，为1，相反为0
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


# Hash值对比
def cmpHash(hash1,hash2):
    n = 0
    # hash长度不同则返回-1代表传参错误
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等n计数+1,n最终为相似度
        if hash1[i] != hash2[i]:
            n = n+1
    return n


if __name__ == '__main__':
    img1 = cv2.imread('lenna.png')
    img2 = Gaussian_noise(Gaussian_noise(img1))
    hash_str_1 = aHash(img1)
    hash_str_2 = aHash(img2)
    print(hash_str_1)
    print(hash_str_2)
    n = cmpHash(hash_str_1,hash_str_2)
    print('均值hash算法相似度：',n)

    hash_str_1 = dHash(img1)
    hash_str_2 = dHash(img2)
    print(hash_str_1)
    print(hash_str_2)
    n = cmpHash(hash_str_1,hash_str_2)
    print('插值hash算法相似度：',n)



