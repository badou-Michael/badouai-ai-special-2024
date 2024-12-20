import numpy as np
import cv2


# 均值hash算法
def mean_hash(img, hash_size=8):
    # 将图片缩放到hash_size大小
    resized = cv2.resize(img, (hash_size, hash_size))
    # 换成灰度图
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # 计算每个像素的平均值  axis=None或不传表示对所有元素求平均值
    mean = np.mean(gray)
    avg = np.sum(mean)
    print('mean: %s' % (avg))
    # 比较平均值，大于平均值为1，小于平均值为0
    hash_value = ''
    for i in range(hash_size):
        for j in range(hash_size):
            if gray[i, j] > avg:
                hash_value += '1'
            else:
                hash_value += '0'
    return hash_value


# 差值hash算法
def difference_hash(img, hash_size=8):
    # 将图片缩放到hash_size大小
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    # 换成灰度图
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    hash_value = ''
    for i in range(hash_size):
        for j in range(hash_size):
            if gray[i, j] > gray[i, j + 1]:
                hash_value += '1'
            else:
                hash_value += '0'
    return hash_value


# 哈希值比较
def hash_compare(hash1, hash2):
    # 比较两个hash值，返回不相同的位数
    if len(hash1) != len(hash2):
        return False
    diff_count = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            diff_count += 1
    return diff_count


# 导入图片
img = cv2.imread('lenna.png')
img2 = cv2.imread('img_binary.png')
# 均值哈希算法
hash_value = mean_hash(img)
hash_value2 = mean_hash(img2)
print(hash_value)
print(hash_value2)

# 差值哈希算法
diff_hash = difference_hash(img)
diff_hash2 = difference_hash(img2)
print(diff_hash)
print(diff_hash2)
# 哈希值比较
compare_result = hash_compare(diff_hash, diff_hash2)

# 输出结果
print(compare_result)
