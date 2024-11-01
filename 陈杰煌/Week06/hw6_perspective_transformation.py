import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

imgdata = img.copy()


# 生成透视变换矩阵；进行透视变换
src = np.float32([[205, 154], [518, 287], [17, 603], [342, 730]])
dst = np.float32([[0, 0], [350, 0], [0, 500], [350, 500]])

# getPerspectiveTransform 函数用于计算透视变换矩阵
m = cv2.getPerspectiveTransform(src, dst)

print("warpMatrix:")

print(m)

# warpPerspective 函数用于进行透视变换
result = cv2.warpPerspective(imgdata, m, (350, 500))

# 显示原图和结果
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
