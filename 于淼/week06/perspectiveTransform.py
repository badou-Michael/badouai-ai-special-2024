import cv2
import numpy as np

img = cv2.imread("F:\DeepLearning\Code_test\photo1.jpg")

# 复制原图像，供后续操作，以免改变原图像
img_copy = img.copy();
print(img.shape)
'''
src——原图像中要截取图片的对应点位置
dst——输出图像中对应的点的位置
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 得到透视变换矩阵

m = cv2.getPerspectiveTransform(src,dst)
print("warpPerspective:")
print(m)

# 将原图片乘以透视变换矩阵，得到Perspective图片
Perspective_img = cv2.warpPerspective(img_copy,m,(337,488))

cv2.imshow("src",img)
cv2.imshow("per_img",Perspective_img)
cv2.waitKey(0)
