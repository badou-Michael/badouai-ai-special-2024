import cv2
import numpy as np

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
print(img)
h, w  = img.shape[:2]
img_eq = np.zeros([h, w], img.dtype)
img_dict = {}
class Index:
    def __init__(self, count, x, y):
        self.count = count
        self.list = []
        self.list.append((x, y))
    def add(self, x, y):
        self.list.append((x, y))
        self.count += 1

for i in range(h):
    for j in range(w):
        if img[i, j] not in img_dict:
            val = Index(1, i, j)
            img_dict[img[i,j]] = val
        else:
            img_dict[img[i, j]].add(i, j)

sumPi = 0
for key in sorted(img_dict.keys()):
    print(f'key:{key} val:{img_dict[key].count}')
    pi  = (img_dict[key].count / (h * w))
    print(f'pi:{pi}')
    sumPi += (img_dict[key].count / (h * w))
    print(f'sumPi: {sumPi}')
    pix = round(sumPi * 256 - 1)
    pix =  0 if pix < 0 else pix

    print(f'pix:{pix}')
    for data in img_dict[key].list:
        img_eq[data[0], data[1]] = pix

print(img_eq)
# img_eq = cv2.equalizeHist(img)
cv2.imshow('original', img)
cv2.imshow('equalized', img_eq)
cv2.waitKey(0)
