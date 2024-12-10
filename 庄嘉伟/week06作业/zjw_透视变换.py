import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
result1 = img.copy()
'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]]) #裁切前的图像对应点位
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])  #裁切后的图像对应点位
print(img.shape)

#生成透视变换矩阵进行透视变换。
m = cv2.getPerspectiveTransform(src, dst)  #获取透视变换矩阵
print("warpMatrix:\n",m)
result = cv2.warpPerspective(result1,m,(337,488)) #透视变换
cv2.imshow("src:",img)
cv2.imshow("result",result)
cv2.waitKey(0)
cv2.destroyWindow()


