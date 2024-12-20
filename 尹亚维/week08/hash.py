import cv2


# 均值hash
def average_hash(img):
    # 缩放为8*8
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ""
    for i in range(8):
        for j in range(8):
            s += gray[i][j]
    # 求均值
    avg = s / 64
    # 得到average_hash
    for i in range(8):
        for j in range(8):
            if gray[i][j] > avg:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


# 差值hash
def difference_hash(img):
    # 使用 8x9 的图像尺寸是为了生成一个 64 位的哈希值
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ""
    for i in range(8):
        for j in range(8):
            if gray[i][j] > gray[i][j + 1]:
                hash_str += "1"
            else:
                hash_str += "0"
    return hash_str


def compare_hash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n


if __name__ == '__main__':
    img1 = cv2.imread('lenna.png')
    img2 = cv2.imread('lenna_gaussian_noise.png')
    hash1 = average_hash(img1)
    hash2 = average_hash(img2)
    n = compare_hash(hash1, hash2)
    print("均值哈希算法相似度：", n)

    hash1 = difference_hash(img1)
    hash2 = difference_hash(img2)
    n = compare_hash(hash1, hash2)
    print("差值哈希算法相似度：", n)
