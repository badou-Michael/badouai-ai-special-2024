'''
直方图均衡化
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 单通道灰度图直方图均衡化
pth = "lenna.png"
img_od = cv2.imread(pth)
img_gray = cv2.cvtColor(img_od, cv2.COLOR_BGR2GRAY)
img_equ = cv2.equalizeHist(img_gray)
# 三通道灰度图直方图均衡化
if img_od.ndim == 3:
    b, g, r = cv2.split(img_od)
img_b = cv2.equalizeHist(b)
img_g = cv2.equalizeHist(g)
img_r = cv2.equalizeHist(r)
img_merged = cv2.merge((img_r, img_g, img_b))

cv2.imshow("Image", img_od)
cv2.imshow("Gray", img_gray)
cv2.imshow("EqualizeHist", img_equ)
cv2.imshow("Merged", cv2.cvtColor(img_merged, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.subplot(141)
plt.axis('off')
plt.title('OD')
plt.imshow(cv2.cvtColor(img_od, cv2.COLOR_BGR2RGB))
plt.subplot(142)
plt.axis('off')
plt.title('GRAY')
plt.imshow(img_gray, cmap='gray')
plt.subplot(143)
plt.axis('off')
plt.title('equ')
plt.imshow(img_equ, cmap='gray')
plt.show()
plt.subplot(144)
plt.axis('off')
plt.title('Merged')
plt.imshow(img_merged)
plt.show()
