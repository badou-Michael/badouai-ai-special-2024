import numpy as np 
import cv2
from skimage import util

img = cv2.imread("/home/zzy/work/lenna.png")
img_noise = util.random_noise(img, mode='gaussian', var = 0.05)
img_noise2 = img_noise.astype(np.float32)
def aHash(img):
    #进行图片缩放8*8
    img = cv2.resize(img, (8,8))

    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #求像素均值
    mean = cv2.mean(gray)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] < mean[0]:
                hash_str += '0'
            else:
                hash_str += '1'
    
    return hash_str

def dHash(img):

    #进行图像缩放 8*9：
    img = cv2.resize(img, (9,8))

    #灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #求差值

    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > gray[i,j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str



def hash_compare(str1, str2):
    if len(str1) != len(str2):
        print("SIZE WRONG!")
        return
    n = 0
    for i in range(len(str1)):
        if str1[i] == str2[i]:
            n += 1
        else:
            continue
    return n
#######test##########

# img = cv2.resize(img, (8,9))
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# print(img_noise2.shape)
# str1 = aHash(img_noise2)

#######end###########
cv2.imshow("original image", img)
cv2.imshow("Gaussian image", img_noise2)
ahashstr1 = aHash(img)
ahashstr2 = aHash(img_noise2)
n = hash_compare(ahashstr1, ahashstr2)
print(len(ahashstr1))
print("均值哈希的相似度：",n)
print("原图像的均值哈希值：",ahashstr1)
print("高斯图像的均值哈希值：",ahashstr2)


dhashstr1 = dHash(img)
dhashstr2 = dHash(img_noise2)
n2 = hash_compare(dhashstr1,dhashstr2)
print("差值哈希的相似度：",n2)
print("原图像的差值哈希值：",dhashstr1)
print("高斯图像的差值哈希值：",dhashstr2)
cv2.waitKey(0)