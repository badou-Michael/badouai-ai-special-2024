"""
yzc  透视变换，先导入图片；根据画图工具获取图像需要变换的顶点像素，以及变换后的尺寸坐标
cv2.getPerspectiveTransform  生成warpMatrix透视变换矩阵
cv2.warpPerspective    生成透视变换图像
"""
import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
result3 = img.copy()

src = np.float32([[207,151],[517,285],[17,601],[343,731]])
dst = np.float32([[0,0],[400,0],[0,500],[400,500]])
matrix = cv2.getPerspectiveTransform(src,dst)
print("warpMatrix: \n",matrix)
result = cv2.warpPerspective(result3,matrix,(400,500))
cv2.imshow("src:",img)
cv2.imshow("dst:",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
