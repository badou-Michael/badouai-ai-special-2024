import cv2
import numpy as np

def aHash(img_1, img_2):
    def aHash_helper(img):
        # 1. 缩放为8*8，保留结构，除去细节
        img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
        # 2.灰度化，转为灰度图
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 3. 求平均值，计算灰度图所有像素的平均值
        avg = np.mean(gray)
        # 4. 比较，像素值大于平均值记作1， 相反记作0，总共64位
        img_hash = ''
        gray_flat = gray.flatten()
        img_hash += ''.join(['0' if i <= avg else '1' for i in gray_flat])
        # 返回hash
        return img_hash
    img1_hash = aHash_helper(img_1)
    img2_hash = aHash_helper(img_2)
    if len(img1_hash) != len(img2_hash):
        return -1
    # 对比hash，计算汉明距离
    hamming_length = sum(1 for i in range(len(img1_hash)) if img1_hash[i] != img2_hash[i])
    return hamming_length

def cmpHash(img_1, img_2):
    def cmpHash_helper(img):
        # 1. 缩放为8*9
        img = cv2.resize(img, (8,9), interpolation=cv2.INTER_CUBIC)
        # 2. 灰度化
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 3. 比较
        gray_flat = gray.flatten()
        img_hash = ''
        img_hash += ''.join(['0' if gray_flat[i] <= gray_flat[i+1] else '1' for i in range(len(gray_flat) - 1)])
        return img_hash
    img1_hash = cmpHash_helper(img_1)
    img2_hash = cmpHash_helper(img_2)
    if len(img1_hash) != len(img2_hash):
        return -1
    # 对比hash，计算汉明距离
    hamming_length = sum(1 for i in range(len(img1_hash)) if img1_hash[i] != img2_hash[i])
    return hamming_length

if __name__ == '__main__':
    # 分别读取原图像和添加高斯噪声的图像，进行对比
    img_org = cv2.imread('lenna.png')
    gaussian_noise = np.random.normal(0, 50, img_org.shape).astype(np.uint8)
    img_noise = cv2.add(img_org,gaussian_noise)

    hash_avg = aHash(img_org, img_noise)
    print('均值哈希算法相似度：',hash_avg)

    hash_cmp = cmpHash(img_org, img_noise)
    print('差值哈希算法相似度：',hash_cmp)


