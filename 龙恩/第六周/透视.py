import cv2
import numpy as np

img=cv2.imread("photo1.jpg")

img1=img.copy()

point1=np.float32([[16,602],[ 207,153],[ 343,732],[519,285]])
point2=np.float32([[0,488],[0,0],[337,488],[337,0]])


m=cv2.getPerspectiveTransform(point1,point2)
result=cv2.warpPerspective(img1,m,(337,488))
cv2.imshow("original",img)
cv2.imshow("after",result)
cv2.waitKey(0)
