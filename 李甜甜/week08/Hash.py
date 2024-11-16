import cv2
# 均值哈希算法 1.读图 2.缩放 3.灰度化 4.求平均值 5.比较大于均值为1，小于为0 。生成哈希
def compute_mean_hash(src_img):
    resize_img = cv2.resize(src_img, (8, 8))
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    #求均值
    sum =0
    for i in range(8):
        for j in range(8):
            sum += gray_img[i, j]
    mean = sum/64
    #比较
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if gray_img[i,j]>mean:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str



#计算差值hash
def compute_difference_hash(src_img):
    resize_img = cv2.resize(src_img,(9,8))
    gray_img = cv2.cvtColor(resize_img,cv2.COLOR_BGR2GRAY)
    hash_str= ''
    for i in range(8):
        for j in range(8):
            if gray_img[i, j] > gray_img[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str

#计算汉明距离
def cmpHash(hash1,hash2):
    n=0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!= hash2[i]:
            n+=1
    return n



img1 = cv2.imread('lenna.png')
img2 = gaussian_blurred_img = cv2.GaussianBlur(img1, (5,5), 0)
hash1= compute_mean_hash(img1)
hash2= compute_mean_hash(img2)
print(hash1)
print(hash2)
n= cmpHash(hash1,hash2)
print('均值哈希算法相似度',n)

#插值哈希算法

img1 = cv2.imread('lenna.png')
img2 = gaussian_blurred_img = cv2.GaussianBlur(img1, (5,5), 0)
hash1= compute_difference_hash(img1)
hash2= compute_difference_hash(img2)
print(hash1)
print(hash2)
n= cmpHash(hash1,hash2)
print('差值值哈希算法相似度',n)
