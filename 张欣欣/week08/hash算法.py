import cv2
# 均值哈希
def aHash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    s = 0
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            # print(gray[i,j])
            s=s+gray[i,j]
    avg = s/64
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                ahash_str=ahash_str+'1'
            else:
                ahash_str=ahash_str+'0'
    return ahash_str

# 差值hash
def dHash(img):
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                dhash_str=dhash_str+'1'
            else:
                dhash_str+='0'
    return dhash_str
# 比较
def compHash(hash1,hash2):
    sum = 0
    if len(hash1)!=len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            sum=sum+1
    return sum

img1 = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_poisson noise.png")
ahash1 = aHash(img1)
ahash2 = aHash(img2)
print("  lenna原图的均值hash值：",ahash1)
print("lenna_noise的均值hash值：",ahash2)
sum_ahash = compHash(ahash1,ahash2)
print("两张图对比后的均值hash值为：",sum_ahash)
print("---------------------------------------------------------------------------")
# img1 = cv2.imread("lenna.png")
# img2 = cv2.imread("lenna_poisson noise.png")
dhash1 = dHash(img1)
dhash2 = dHash(img2)
print("  lenna原图的差值hash值 ：",dhash1)
print("lenna_noise 的差值hash值：",dhash2)
sum_dhash = compHash(dhash1,dhash2)
print("两张图对比后的差值hash值为：",sum_dhash)
