from skimage.color import rgb2gray
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#灰度化
sourcepic = cv2.imread("test.jpg")
rows, cols = sourcepic.shape[:2]
graypic = np.zeros([rows, cols], sourcepic.dtype)
for i in range(rows):
    for j in range(cols):
        pos = sourcepic[i, j]
        graypic[i, j] = int(pos[0]*0.11+pos[1]*0.59+pos[2]*0.3)
plt.subplot(221)
pic221 = plt.imread("test.jpg")
plt.imshow(pic221)
graypic = rgb2gray(sourcepic) # OR graypic = cv2.cvtColor(sourcepic, cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(graypic, cmap='gray')
#二值化
binarypic = np.where(graypic >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(binarypic, cmap='gray')
plt.show()
