#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
'''

img = cv2.imread("../sea.jpg", 1)

# 获取显示器分辨率
screen_width = 2560
screen_height = 1440


# 调整图像大小，保持纵横比
aspect_ratio = img.shape[1] / img.shape[0]
if aspect_ratio > 1:  # 宽大于高
    new_width = screen_width
    new_height = int(screen_width / aspect_ratio)
else:  # 高大于宽
    new_height = screen_height
    new_width = int(screen_height * aspect_ratio)

# 调整图像大小
resized_image = cv2.resize(img, (new_width, new_height))

gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 53, 245)

# 灰度图和边缘检测图水平拼接
combined = cv2.hconcat([gray, edges])
cv2.imshow("canny", combined)
cv2.waitKey()
cv2.destroyAllWindows()
