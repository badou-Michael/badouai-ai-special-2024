import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
imgcopy = img.copy()
l,h=img.shape[:2]
#输出图像坐标点及转换后坐标点
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
#获取转换矩阵:cv2.getPerspectiveTransform()
m = cv2.getPerspectiveTransform(src, dst)
#透视变换:cv2.warpPerspective(原图,转换矩阵, (转换后大小))
all = cv2.warpPerspective(imgcopy, m,[h,l])     #整体图
part = cv2.warpPerspective(imgcopy, m, [337, 488]) #局部图
imgs=np.hstack([img,all])

cv2.imshow("src", imgs)
cv2.imshow("result", part)
cv2.waitKey()
