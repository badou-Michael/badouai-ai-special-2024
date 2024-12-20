import cv2

def DHash(img):

    #缩放，保留结构，去除细节
    sca_img = cv2.resize(img,(8,9))

    #灰度化
    img_gray = cv2.cvtColor(sca_img,cv2.COLOR_BGR2GRAY)

    #比较
    Hash_char = ''
    H, W = img_gray.shape
    for h in range(H):
        for w in range(W-1):
            if img_gray[h,w] > img_gray[h,w+1]:
                Hash_char += '1'
            else:
                Hash_char += '0'
    return Hash_char

#对比指纹，计算汉明距离
def Compere_Hash(HD1,HD2):
    if len(HD1) != len(HD2):
        return -1

    sum = 0

    for i in range(len(HD1)):
        if HD1[i] != HD2[i]:
            sum += 1

    return sum

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_noise = cv2.imread('lenna_noise.png')

    Hash_1 = DHash(img)
    Hash_2 = DHash(img_noise)

    print('不同指纹个数为:',Compere_Hash(Hash_1,Hash_2))
