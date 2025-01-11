# # Canny是目前最优秀的边缘检测算法之一，其目标为找到一个最优的边缘，其最优边缘的定义为：
# 1、好的检测：算法能够尽可能的标出图像中的实际边缘
# 2、好的定位：标识出的边缘要与实际图像中的边缘尽可能接近
# 3、最小响应：图像中的边缘只能标记一次

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
主要参数说明
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图像
第二个参数是阈值1
第三个参数是阈值2
'''

import cv2 
import numpy as np

def CannyThreshold(lowThreshold):  
    detected_edges = cv2.Canny(gray,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #边缘检测

    dst = cv2.bitwise_and(img, img, dst=None, mask = detected_edges)  
    cv2.imshow('Canny result',dst)   
      
'''
bitwise_and 函数
cv2.bitwise_and 是 OpenCV 提供的一个函数,用于对两个图像进行逐位“与”操作。其函数签名如下：
cv2.bitwise_and(src1, src2, dst=None, mask=None)
src1 和 src2: 这是两个输入图像,它们必须具有相同的大小和类型。
dst: 这是输出图像,可以省略,函数会自动创建一个与输入图像大小和类型相同的图像。
mask: 这是一个可选的掩码图像。只有在掩码图像中对应位置的像素值不为零时,才会对 src1 和 src2 的对应位置的像素进行“与”操作。
mask 掩码
掩码是一个二值图像(即每个像素值要么是0,要么是1),用于指定哪些像素应该参与操作。
在 cv2.bitwise_and 中,只有在掩码图像中对应位置的像素值为1时,才会对 src1 和 src2 的对应位置的像素进行“与”操作。
掩码图像的大小和类型必须与输入图像相同。

detected_edges 检测到的边缘
detected_edges 是通过边缘检测算法（如 Canny 边缘检测）得到的边缘图像。
这个图像通常是一个二值图像,其中边缘像素的值为1 (或255) ,非边缘像素的值为0。
在你的代码中,detected_edges 被用作掩码,来保留原始图像中对应边缘位置的像素值。
代码解释
dst = cv2.bitwise_and(img, img, mask=detected_edges)  
cv2.imshow('Canny result', dst)

这段代码的作用是：

使用 detected_edges 作为掩码,对原始图像 img 进行逐位“与”操作。
只有在 detected_edges 中对应位置的像素值为1时,结果图像 dst 中的对应位置才会保留原始图像 img 的像素值。
使用 cv2.imshow 函数显示结果图像 dst。
这段代码的目的是将检测到的边缘保留在结果图像中,而其他非边缘部分将被过滤掉。
'''
    
lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

# 读取图像
img = cv2.imread('lenna.png')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #转换彩色图像为灰度图像

# 创建窗口
cv2.namedWindow('Canny result')  

# 创建滑动条
'''
以下是创建滑动条的cv2.createTrackbar()函数
有5个参数, 分别解释如下: 
第一个参数是滑动条的名称
第二个参数是滑动条所属窗口的名称
第三个参数是滑动条的默认值
第四个参数是滑动条的最大值(0~count)
第五个参数是滑动条变化时调用的回调函数
'''
cv2.createTrackbar('Min threshold', 'Canny result', lowThreshold, max_lowThreshold, CannyThreshold)

# 初始化
CannyThreshold(0)  # initialization  
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()