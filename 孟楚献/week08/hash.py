import cv2
import numpy as np
from PIL.ImImagePlugin import number


def average_hash(img_path)->str:
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hash = cv2.resize(img_gray, (8, 8))
    img_hash = img_hash.flat
    ave = np.average(img_hash)
    img_hash = img_hash > ave
    res = ""
    for i in range(0, len(img_hash)):
        res += str(number(img_hash[i]))
    return res

def diff_hash(img_path)->str:
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 先宽度，后高度
    img_hash = cv2.resize(img_gray, (9, 8))
    res = ""
    for row in range(0, img_hash.shape[0]):
        for col in range(1, img_hash.shape[1]):
            res += str(number(img_hash[row, col] > img_hash[row, col - 1]))
    return res

def hash_comparison(hash_1: str, hash_2: str)->number:
    if len(hash_1) != len(hash_2):
        raise "哈希值长度不同！"
    res = 0
    for i in range(0, len(hash_1)):
        res += number(hash_1[i] != hash_2[i])
    print("两个哈希值的汉明距离为", res, "相似度为", 1 - res / len(hash_1))
    return res

if __name__ == "__main__":
    # hash_1 = average_hash("../lenna.png")
    # hash_2 = average_hash("../lennaBilinear.png")
    hash_1 = average_hash("../th.jfif")
    hash_2 = average_hash("../thBilinear.png")
    hash_comparison(hash_1, hash_2)

    # hash_1 = diff_hash("../lenna.png")
    # hash_2 = diff_hash("../JJ.png")
    hash_1 = diff_hash("../th.jfif")
    hash_2 = diff_hash("../thBilinear.png")
    hash_comparison(hash_1, hash_2)
