import cv2
import numpy as np

img=cv2.imread("lenna.png")#默认1  BGR
print(img.shape)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("canny",cv2.Canny(gray,150,300))#参数2为低阈值，参数3为高阈值
cv2.waitKey()#无限延时  
cv2.destroyAllWindows()#关闭窗口
