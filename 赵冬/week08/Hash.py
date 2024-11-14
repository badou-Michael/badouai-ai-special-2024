import cv2
import numpy as np

def avg_hash(img):
    img = cv2.resize(img, (8,8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = np.mean(gray)
    z = np.zeros(shape=gray.shape, dtype=np.uint8)
    z[gray>m]=1
    z = z.flatten()
    d = ""
    for i in z:
        d += str(i)
    return d

def diff_hash(img):
    img = cv2.resize(img, (9,8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(int)
    diff = np.diff(gray).flatten()
    z = np.zeros(shape=diff.shape, dtype=np.uint8)
    z[diff<0]=1
    d = ""
    for i in z:
        d += str(i)
    return d

if __name__ == '__main__':
    lenna = cv2.imread("lenna.png")
    aHash = avg_hash(lenna)
    print("均值Hash")
    print(aHash)
    dHash = diff_hash(lenna)
    print("差值Hash")
    print(dHash)
