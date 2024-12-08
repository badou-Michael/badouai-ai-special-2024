def aHash(img):
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sum_pixel=0
    hash_str=''
    
    for i in range(8):
        for j in range(8):
            sum_pixel+=gray[i,j]
    
    avg_pixel=sum_pixel/64
    
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg_pixel:
                hash_str+='1'
            else:
                hash_str+='0'
    
    return hash_str


def dHash(img):
    img=cv2.resize(img,(8,9),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    hash_str=''
    
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str+='1'
            else:
                hash_str+='0'
                
    return hash_str


def cmpHash(hash1,hash2):
    assert len(hash1)==len(hash2)
    n=0
    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n+=1
    return n


img1=cv2.imread('lenna.png')
img2=cv2.imread('lenna_noise.png')
hash1= aHash(img1)
hash2= aHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('均值哈希算法相似度：',n)
 
hash1= dHash(img1)
hash2= dHash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print('差值哈希算法相似度：',n)
