#1.二值化
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
img = cv2.imread("E:\work_soft\lenna.png")
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i,j] <=0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i, j] = 1
print("-----imge_binary------")
print(img_gray)
print(img_gray.shape)

plt.subplot(223)
plt.imshow(img_gray, cmap='gray')
plt.show()

#2.二值化
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
img = cv2.imread("E:\work_soft\lenna.png")
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()

#3.二值化
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
img = cv2.imread("E:\work_soft\lenna.png")
img_gray =rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
img_binary = rgb2gray(img)
rows, cols = img_binary.shape
for i in range(rows):
    for j in range(cols):
        if img_binary[i,j] <=0.5:
            img_binary[i,j] = 0
        else:
            img_binary[i, j] = 1
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
