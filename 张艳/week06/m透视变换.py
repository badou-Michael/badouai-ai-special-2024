import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]]) #原图坐标（依次对应）
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]]) #新图坐标
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst) # @wrapMatrix.py
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)

''' result3 = img.copy()
result3 = img.copy() 创建了一个独立的副本，两者互不影响。
result4 = img 只是创建了一个引用，两者指向同一个对象，修改其中一个会影响另一个。
'''
