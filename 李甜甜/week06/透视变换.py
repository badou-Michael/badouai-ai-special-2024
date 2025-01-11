import cv2
import numpy as np
#读取图片
img = cv2.imread("photo1.jpg")
#复制一个图片用作装换后的图
result3 = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
#得到透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result",result)
cv2.waitKey(0)
