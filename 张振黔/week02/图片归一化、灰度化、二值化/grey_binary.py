from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#灰度化
image=cv2.imread('lenna.png')
image_normalized = image / 255.0 #归一化
h, w = image.shape[:2]#获取图像矩阵前两个参数，即高和宽，第三个参数为通道
image_grey=np.zeros([h,w],image_normalized.dtype)
for i in range(h):
    for j in range(w):
        BGR=image_normalized[i,j]
        image_grey[i,j]=(BGR[0]*0.11+BGR[1]*0.59+BGR[2]*0.3) #B G R=0.11 0.59 0.3
#二值化
image_binary=np.zeros([h,w],float)
for i in range(h):
    for j in range(w):
        if image_grey[i,j]>=0.54: #若未归一化，二值点为 0，255；调整该值能够影响黑白图比重
            image_binary[i,j]=1
        else:image_binary[i,j]=0
#生成矩阵
#print("---image---\n%s" % image)
print("---image show Normalized---\n%s" % image_normalized)
print("---image show gray---\n %s" % image_grey)
print("---image show binary---\n%s" % image_binary)
#显示图像
#cv2.imshow("image show gray", image)
cv2.imshow("image show Normalized", image_normalized)
#cv2.imshow("image show gray", image_grey)
#cv2.imshow("image show gray", image_binary)
#cv2.waitKey(0)
imgs = np.hstack([image_grey,image_binary])
cv2.imshow("All image", imgs)
cv2.waitKey()
cv2.destroyAllWindows()
