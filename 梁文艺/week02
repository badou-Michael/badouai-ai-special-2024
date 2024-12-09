
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

tu =cv2.imread("E:\lenna.png")
l,w=tu.shape[:2]
diban=np.zeros([l,w],tu.dtype)
for i in range(l):
      for j in range(w):
            xsd =tu[i,j]
            diban[i,j]= int(xsd[0]*0.11+xsd[1]*0.59+xsd[2]*0.3)

print(xsd)
print(diban)
print("tu show gray:%s"%diban)
cv2.imshow("tu show gray",diban)

plt.subplot(221)
tu =plt.imread("E:\lenna.png")
plt.imshow(tu)
print("image lenna")
print(tu)

diban=rgb2gray(tu)
plt.subplot(222)
plt.imshow(diban,cmap='gray')
print("tu gray")
print(diban)

tu_binary = np.where( diban >= 0.5,1,0)
print("tu")
print(tu_binary)
print(tu_binary.shape)
plt.subplot(223)
plt.imshow(tu_binary,cmap='gray')
plt.show()





















