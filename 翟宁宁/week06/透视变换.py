import  cv2
import numpy as np

img =  cv2.imread('photo1.jpg')

src = np.float32([[153,206],[286,517],[604,15],[732,343]])
dis = np.float32([[0,0],[512,0],[0,512],[512,512]])

print('src_img------------- \n',img)
#使用OpenCV的接口 ，获取透视变换矩阵,由上面的四对点，确定一个对应关系
warpMatrix = cv2.getPerspectiveTransform(src,dis)
print('warpMatrix------------- \n',warpMatrix)
#使用warpMatrix 把所有的点映射完

res = cv2.warpPerspective(img,warpMatrix,(512,512))
cv2.imshow('src',img)
cv2.imshow('透视结果',res)
cv2.waitKey(0)
