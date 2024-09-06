import cv2

# #############################  CV2的读取和显示，imread、imshow ############################
# OpenCV 对图像的任何操作，本质上就是对 Numpy 多维数组的运算
# 结果：手动关了img1后，显示img2
############################### 例1.简单显示原图和灰度图 #####################################
'''
# 读取原图并以BGR模式显示
img = cv2.imread("lenna.png")  # 返回该图像的矩阵, nparray 多维数组
# print("___________original matrix:________\n"+img)
cv2.imshow("Original image", img)
# imshow之后跟waitkey，表示无限循环show，等待用户按键触发跳出循环
key = cv2.waitKey(0)

# 【显示灰度图法一】: imread的flags=0 (默认flags=1 BGR)
img2 = cv2.imread("lenna.png", flags=0)
cv2.imshow("Gray image", img2)
key = cv2.waitKey(0)
'''
############################### 例2. 多个图像组合显示，用np.hstack#####################################
'''
import numpy as np
img1 = cv2.imread("lenna.png")

img2 = cv2.imread("lenna.png", flags=0)

# error! 三通道和单通道的图并列失败，因为数组维度不同。(第二个图不应该是1个维度么？) img2也flag1就能显示
imgStack = np.hstack((img1, img2)) # ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 3 dimension(s) and the array at index 1 has 2 dimension(s)
cv2.imshow("original & gray", imgStack)
key = cv2.waitKey(0)
'''
############################### 例3. 多个图像组合显示，用plt#####################################
'''
import matplotlib.pyplot as plt
img1 = cv2.imread("lenna.png")

# 使用plt show image(RGB格式)，需要先把cv的BGR格式转换
# cvtColor：https://blog.csdn.net/u011775793/article/details/134777165
img1RGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# subplots()：用于创建子图(行列编号), plt.title("BGR image"):图片命名，plt.axis('off')：不显示坐标
# subplot(nrows, ncols, index, **kwargs)
# plt.subplot(221)
plt.subplot(221), plt.title("Original image"), plt.axis('off')
# 创建一个图像对象
plt.imshow(img1RGB)

plt.subplot(222), plt.title("BGR image"), plt.axis('off')
plt.imshow(img1)

# 【显示灰度图法二】: cv2.COLOR_BGR2GRAY，需要使用 cmap=‘gray’ 进行参数设置。
# img2 = cv2.imread("lenna.png", flags=0)
img1GRAY = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.subplot(223), plt.title("Gray image"), plt.axis('off')
plt.imshow(img1GRAY, cmap='gray')

plt.subplot(224), plt.title("Gray image：no cmap programs"), plt.axis('off')
plt.imshow(img1GRAY)

# 显示所有已创建的图形
plt.show()
'''
############################### 例4. 灰度图法三(skimage)、法四(numpy)/plt read img #####################################
'''
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 显示原图
img1 = plt.imread("lenna.png")
plt.subplot(221), plt.title("Original image"), plt.axis('off')
# 创建一个图像对象
plt.imshow(img1)

# 显示灰度图【法三】：skimage.color.rgb2gray
img2 = rgb2gray(img1)
plt.subplot(222), plt.title("Gray image using skimage"), plt.axis('off')
plt.imshow(img2, cmap='gray')

import numpy as np

# 显示灰度图【法四】：plt read的灰度原理计算
########debug###########
# print(img1)  # plt read的原图的三维范围是[0-1],用整数方法
#
# img_cv2 = cv2.imread("lenna.png")
# img_cv2_RGB = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
# print(img_cv2_RGB)  # cv2 read的原图的三维范围是[0-255]，用浮点算法
########debug###########

h, w = img1.shape[:2]
img3 = np.zeros([h, w], dtype=img1.dtype) # img3 = np.zeros([h, w], img1.dtype) 一样的
# print("aaa:"+str(img1.ndim == 3)) #True
# print("bbb:"+str(img1.shape[2] == 3)) #True
for i in range(h):
    for j in range(w):
        m = img1[i, j]
        img3[i, j] = int(m[0]*30 + m[1]*59 + m[2]*11)

print(img3)

plt.subplot(223), plt.title("Gray image using principle(plt read)"), plt.axis('off')
plt.imshow(img3, cmap='gray')
plt.show()
'''
############################### 例4. 灰度图法五(numpy)/cv2 read#####################################
# '''
# 显示灰度图【法五】：cv2 read的灰度原理计算
import cv2
import numpy as np

# img = cv2.imread("lenna.png")
img = cv2.imread("lenna.png")
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # all
h, w = img.shape[:2] #.shape

img_Gray = np.zeros([h, w], img_RGB.dtype)  # all
# img_Gray_average = img_Gray # error! 这一步是赋地址，会让三个灰度图像都是img_Gray的值
# img_Gray_Green = img_Gray
# img_Gray_Zero = img_Gray
img_Gray_average =  np.zeros([h, w], img_RGB.dtype)
img_Gray_Green =  np.zeros([h, w], img_RGB.dtype)
img_Gray_Zero = np.zeros([h, w], img_RGB.dtype)
img_Gray2 = np.zeros([h, w], img.dtype)  # all
for i in range(h):
    for j in range(w):
        m = img_RGB[i, j]
        img_Gray[i, j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)
        # 不转RGB：
        # m = img[i, j]
        # img_Gray2[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

        # 【其他灰度图计算公式】
        img_Gray_average[i, j] = int((m[0] + m[1] + m[2])/3) # RuntimeWarning: overflow encountered in ubyte_scalars # 是因为相加后矩阵范围不是0-255了，
        img_Gray_Green[i, j] = m[1]

# cv2.imshow("img_Gray", img_Gray)
# cv2.waitKey(0)
# # cv2.imshow("img_Gray2", img_Gray2)
# # cv2.waitKey(0)
# # 转RGB和不转RGB用plt显示灰度图的效果不一样，print出来结果也不一样
# print(img_Gray)
# print(img_Gray2)
# np.set_printoptions (threshold=np.inf)
print("img_Gray:{}".format(img_Gray))
print("img_Gray_Green:{}".format(img_Gray_Green))
print("img_Gray_average:{}".format(img_Gray_average))
# print(img_Gray_Green == img_Gray)
# print(img_Gray_average == img_Gray)

# 以下是各个灰度值算法，以及plt cmap用法的研究
import matplotlib.pyplot as plt
plt.subplot(221), plt.title("Orignal image"), plt.axis('off')
plt.imshow(img_RGB) # 默认颜色映射是None
# plt.imshow(img_RGB, cmap=None) # 原图
# plt.imshow(img_RGB, cmap="gray") # 原图
# plt.imshow(img) # BGR图，偏紫偏蓝

plt.subplot(222), plt.title("Gray image in average"), plt.axis('off')
plt.imshow(img_Gray_average, cmap='gray')

plt.subplot(223), plt.title("Gray image in green"), plt.axis('off')
plt.imshow(img_Gray_Green, cmap='gray')
# cmap参数作用：给图像上色, 在灰度图中要加上gray参数，其他原图和白图中加不加没啥影响(?)
# plt.subplot(223), plt.title("Gray image in zero"), plt.axis('off')
# plt.imshow(img_Gray_Zero) # 白图
# plt.imshow(img_Gray_Zero, cmap='gray') # 白图

plt.subplot(224), plt.title("Gray image"), plt.axis('off')
plt.imshow(img_Gray, cmap='gray')
# plt.imshow(img_Gray)

# 遗留：为什么三种计算方法的灰度图效果一样？print结果也一样？
plt.show()
# '''
############################### 例5. 二值化计算#####################################
"""
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

img = plt.imread("lenna.png")
plt.subplot(221), plt.title("Original image"), plt.axis('off')
plt.imshow(img)

# 先灰度化
img_gray = rgb2gray(img)
plt.subplot(222), plt.title("Gray image"), plt.axis('off')
plt.imshow(img_gray, cmap='gray')

# 【二值化计算法一：使用np.where】
h, w = img_gray.shape

img_binary1 = np.where(img_gray >= 0.5, 1, 0)

plt.subplot(223), plt.title("Binary image by np.where"), plt.axis('off')
plt.imshow(img_binary1, cmap='gray')

# 【二值化计算法二：手动计算】
h, w = img_gray.shape
img_binary2 = np.zeros([h, w], img_gray.dtype)
for i in range(h):
    for j in range(w):
        if (img_gray[i, j] <= 0.5):
            img_binary2[i, j] = 0
        else:
            img_binary2[i, j] = 1

plt.subplot(224), plt.title("Binary image by hand"), plt.axis('off')
plt.imshow(img_binary2, cmap='gray')


# 【二值化计算法三：用cv2公式】
# https://blog.csdn.net/fujian87232/article/details/115712763
import cv2
# retval, dst = cv.threshold( src, thresh, maxval, type[, dst] )
# 直接使用plt的read和灰度矩阵
# ret, img_binary3 = cv2.threshold(img_gray, 0.5, 1, cv2.THRESH_BINARY)
# cv2的read
img_cv2 = cv2.imread("lenna.png")
img_gray2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
ret, img_binary3 = cv2.threshold(img_gray2, 127, 255, cv2.THRESH_BINARY)  # all

plt.subplot(224), plt.title("Binary image by cv2 THRESH_BINARY"), plt.axis('off')
plt.imshow(img_binary3, cmap='gray')

plt.show()
"""
