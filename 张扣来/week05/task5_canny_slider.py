import cv2
import numpy as np

'''
Canny 边缘检测，程序优化
'''
img = cv2.imread('../../../request/task2/lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def CannyThreshold (lowThreshold):
    # 边缘检测
    '''
    cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges
    image：输入图像，必须为单通道灰度图像；
    threshold1：第一个阈值，用于边缘连接；
    threshold2：第二个阈值，用于边缘检测；
    edges：输出的边缘图像；
    apertureSize：Sobel 算子的大小，可选值为 3、5、7，默认值为 3；
    L2gradient：是否使用 L2（2是下标）范数计算梯度大小，可选值为 True 和 False，默认值为 False。
    cv2.Canny() 函数的返回值为边缘图像。
    注：第一个阈值参数为低阈值，用于确定哪些梯度变化被认为是潜在的边缘。所有梯度值高于低阈值的像素点都被认为是潜在的边缘点。
    第二个阈值参数为高阈值，用于确定哪些潜在的边缘点是真正的边缘。所有梯度值高于高阈值的像素点都被认为是真正的边缘点。
    同时，所有梯度值低于低阈值的像素点都被认为不是边缘点。在实际应用中，合适的阈值参数需要根据具体图像和任务进行调整，以获得最佳效果。
    通常，可以通过试验不同的参数值来确定最佳的阈值参数。
    '''
    detected_edges = cv2.Canny(gray,lowThreshold,lowThreshold*ratio,apertureSize= kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)
    cv2.imshow('canny_slider',dst)
    # '''
    # cv2.bitwise_and()函数用于执行图像的按位与操作
    # 按位与操作是指对两幅图像的像素进行逐位比较，当且仅当两幅图像的对应像素值都为1时，结果图像的对应像素值才为1；
    # 否则为0。这种操作常用于图像融合、掩码操作等场景。
    # cv2.bitwise_and(src1, src2, dst=None, mask=None)
    # src1：第一幅输入图像
    # src2：第二幅输入图像
    # dst：可选参数，输出图像，与输入图像具有相同的尺寸和数据类型
    # mask：可选参数 ，掩码图像，用于指定哪些像素进行按位与操作
    # 输入图像的尺寸和数据类型必须相同，否则会导致错误。
    # 可以使用掩码图像来指定哪些像素进行按位与操作。
    # '''
lowThreshold = 0
max_lowThreshold = 300
ratio = 3
kernel_size = 3
# 创建窗口
cv2.namedWindow('canny_slider')
cv2.createTrackbar('slider_range','canny_slider',lowThreshold,max_lowThreshold,CannyThreshold)
cv2.createTrackbar('slider_range2','canny_slider',300,500,CannyThreshold)
# 调用函数
CannyThreshold(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象放置在哪个窗口的名称
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''