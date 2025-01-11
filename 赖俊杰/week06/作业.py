import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)

m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', 0) 
print (img.shape)

titles = [u'原始图像', u'聚类图像']  
images = [img, dst]  
for i in range(2):  
   plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]), plt.yticks([])  
plt.show()
