import numpy as np
import cv2


def averageHash(imgdata):
    """
    1. 缩放：图片缩放为8*8，保留结构，除去细节。
    2. 灰度化：转换为灰度图。
    3. 求平均值：计算灰度图所有像素的平均值。
    4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
    5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不
    相同位数越少，图片越相似。
    @param data:
    @return:
    """
    # 缩放：图片缩放为8*8，保留结构，除去细节,
    # INTER_CUBIC:双三次插值，考虑周围16个像素的像素值，适用于高质量图像处理
    imagenew = cv2.resize(imgdata, dsize=(8, 8), interpolation=cv2.INTER_CUBIC)
    # 灰度化
    imageGray = cv2.cvtColor(imagenew, cv2.COLOR_BGR2GRAY)
    # 求平均值
    meandata = np.mean(imageGray)
    # 比较，生成哈希字符串
    hashStr = ''
    # 比较：像素值大于平均值记作1，相反记作0，总共64位。
    for i in range(8):
        for j in range(8):
            if imageGray[i, j] > meandata:
                hashStr = hashStr + '1'
            else:
                hashStr = hashStr + '0'

    return hashStr


def diffHash(imgdata):
    """
    差值哈希：
    1. 缩放：图片缩放为8*9，保留结构，除去细节。
    2. 灰度化：转换为灰度图。
    3. 求平均值：计算灰度图所有像素的平均值。 ---这步没有，只是为了与均值哈希做对比
    4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，
    八个差值，有8行，总共64位
    5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样
    的，不相同位数越少，图片越相似。
    @param imgdata: 图像数据
    @return:
    """
    #缩放：图片缩放为8*9，保留结构，除去细节。
    #缩放8*9
    resizeimg = cv2.resize(imgdata,dsize=(9,8),interpolation=cv2.INTER_CUBIC)
    print(resizeimg)
    #灰度化：转换为灰度图
    gray_img = cv2.cvtColor(resizeimg,cv2.COLOR_BGR2GRAY)
    hashstr = ""
    for i in range(8):
        for j in range(8):
            if gray_img[i,j]>gray_img[i,j+1]:
                hashstr = hashstr+"1"
            else:
                hashstr = hashstr + "0"
    return hashstr

def compareHamminDistince(data1,data2):
    """
    比较汉明距离，汉明距离长度要想等
    @param data1:
    @param data2:
    @return:
    """
    if len(data1)!=len(data2):
        return -1
    num = 0
    for i in range(len(data1)):
        if data1[i]!=data2[i]:
            num = num+1
    return num

if __name__ == '__main__':
    normalimg = cv2.imread("lenna.png")
    noiseimg = cv2.imread("noise_image.jpg")
    ahash_normal = averageHash(normalimg)
    ahash_noise = averageHash(noiseimg)
    A_distance = compareHamminDistince(ahash_normal,ahash_noise)
    dhash_normal = diffHash(normalimg)
    dhash_noise = diffHash(noiseimg)
    D_distance = compareHamminDistince(dhash_normal,dhash_noise)
    print("ahash_normal\n", ahash_normal)
    print("ahash_noise\n", ahash_noise)
    print('均值哈希算法相似度：',A_distance)
    print('差值哈希算法相似度：',D_distance)