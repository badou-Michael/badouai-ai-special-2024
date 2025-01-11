import cv2
import numpy as np

img1=cv2.imread('lenna.png')              #原图
img2=cv2.imread('lenna_noise.png')        #加上噪声的图

def junzhiHash(image):
  img = cv2.resize(image, (8,8), interpolation = cv2.INTER_CUBIC)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #求平均灰度
  total_gray = np.sum(gray_image)
  avg_gray = total_gray / 64

  hash_str = ''
  for i in range(8):
    for j in range(8):
      if gray_image[i,j] > avg_gray:
        hash_str = hash_str + '1'
      else:
        hash_str = hash_str + '0'
    return hash_str

def chazhiHash(image):
  img = cv2.resize(image, (8,9), interpolation = cv2.INTER_CUBIC)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  hash_str = ''
  for i in range(8):
    for j in range(8):
      if gray_image[i,j] > gray_image[i,j+1]:
        hash_str = hash_str + '1'
      else:
        hash_str = hash_str + '0'
  return hash_str

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

img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna_noise.png')
hash1= junzhiHash(img1)
hash2= junzhiHash(img2)

print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('均值哈希算法相似度：',n)

img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna_noise.png')
hash3= chazhiHash(img1)
hash4= chazhiHash(img2)

print(hash3)
print(hash4)
n=cmpHash(hash3,hash4)
print('均值哈希算法相似度：',n)
