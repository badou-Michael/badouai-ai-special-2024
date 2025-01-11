import cv2

img = cv2.imread("d:\\Users\ls008\Desktop\lenna.png",0)

def avgHash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s=0
    hash_img = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i,j]
    s_avg = s/(8*8)
    for i in range(8):
        for j in range(8):
            if gray[i,j] > s_avg:
                hash_img = 1
            else:
                hash_img = 0
    return hash_img

def diffHash(img):
    img = cv2.resize(img,(8,9),interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    s=0
    hash_img = ''
    for i in range(8):
        for j in range(9):
            s = s + gray[i,j]
    s_avg = s/64
    for i in range(8):
        for j in range(8):
            if gray[i,j] > [i,j+1]:
                hash_img = 1
            else:
                hash_img = 0
    return hash_img

def compare(hash1,hash2):
    result = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            result = result+1
    return result


img1 = img
img2 = img
hash1 = avgHash(img1)
hash2 = avgHash(img2)
print(hash1)
print(hash2)
n = compare(hash1, hash2)
print('均值哈希算法相似度：', n)

hash1 = img
hash2 = img
print(hash1)
print(hash2)
n = compare(hash1, hash2)
print('差值哈希算法相似度：', n)
