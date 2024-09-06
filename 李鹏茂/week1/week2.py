import  numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread(r'C:\Users\Lenovo\Desktop\meinv.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]
print(h)
print(w)
image_gray  = np.zeros((h,w),img.dtype)
image_gray[:] = 0.5*img[:,:,0] + 0.13 * img[:,:,1] + 0.07 * img[:,:,2]
print(image_gray)


image_BIN  = np.where(image_gray>=10,1,0)# 绘制图像
print(image_BIN)
plt.figure(figsize=(10, 7))

plt.subplot(221)
plt.imshow(img)
plt.title('Original RGB Image')
plt.axis('off')  # 不显示坐标轴

plt.subplot(222)
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')  # 不显示坐标轴

plt.subplot(223)
plt.imshow(image_BIN, cmap='gray')
plt.title('Binary Image')
plt.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.show()