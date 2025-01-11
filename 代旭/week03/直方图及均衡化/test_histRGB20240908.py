import cv2
import matplotlib.pyplot as plt



img = cv2.imread("lenna.png",0)
equalizeHist_cv2 = cv2.equalizeHist(img)
img1 = cv2.imread("lenna.png")
(src_b,src_g,src_r)=cv2.split(img1)
dst_b = cv2.equalizeHist(src_b)
dst_g = cv2.equalizeHist(src_g)
dst_r = cv2.equalizeHist(src_r)
dstImg = cv2.merge((dst_b,dst_g,dst_r))


plt.subplot(221)
plt.imshow(img,cmap='gray')
plt.title('Original')
plt.subplot(222)
plt.imshow(equalizeHist_cv2,cmap='gray')
plt.title('Equalized')
plt.subplot(223)
plt.imshow(img1)
plt.subplot(224)
plt.imshow(dstImg)

plt.show()
