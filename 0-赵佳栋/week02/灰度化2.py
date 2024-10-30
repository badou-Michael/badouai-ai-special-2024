'''
@Project ：BadouCV 
@File    ：test02.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/10 15:26 
'''
import cv2

# 读取彩色图像
img_bgr = cv2.imread("../lenna.png")

# 使用 OpenCV 提供的函数直接转换
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 也可以使用 skimage.color 库的函数rgb2gray
#result_img = rgb2gray(img)

# 显示灰度图像
cv2.imshow("Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
