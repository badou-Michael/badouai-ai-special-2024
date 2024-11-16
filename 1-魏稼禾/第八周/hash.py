import cv2
import numpy as np

def hammin_dist(hash1, hash2):
    assert len(hash1) == len(hash2)
    return np.sum([0 if i==j else 1 for i,j in zip(hash1, hash2)])

# 均值hash
def avg_hash(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (8,8))
    avg_pix = np.mean(img)
    img = np.reshape(img, (-1))
    hash = [1 if i > avg_pix else 0 for i in img]
    return hash

# 差值hash
def diff_hash(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (9,8))    # resize(img, (width,height))   和np中的行列是反的
    hash = np.zeros((8,8))
    for r in range(8):
        for c in range(8):
            if img[r][c] > img[r][c+1]:
                hash[r][c] = 1
    return np.reshape(hash, (-1))

if __name__ == "__main__":
    avg_hash1 = avg_hash("1.jpg")
    avg_hash2 = avg_hash("2.jpg")
    print(hammin_dist(avg_hash1, avg_hash2))