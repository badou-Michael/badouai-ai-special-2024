import cv2

ori_img = cv2.imread("lenna.png")

# 高斯滤波
h_img = cv2.GaussianBlur(ori_img,(7,7),2)
# 均值滤波
h_img = cv2.blur(ori_img,(7,7))
# 中值滤波
h_img = cv2.medianBlur(ori_img,5)
# 双边滤波
h_img = cv2.bilateralFilter(ori_img,20,120,300)