import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../../../request/task2/lenna.png',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ksize=3表示3*3kernel进行处理
# 边缘幅度算法：sobel，pwitt，laplace，Canny算子
img_sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
img_sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
print('卷积之后谁水平方向的数据：\n',img_sobel_x)
img_sobel_x_abs = cv2.convertScaleAbs(img_sobel_x)
img_sobel_y_abs = cv2.convertScaleAbs(img_sobel_y)
print('卷积之后的水平方向数据负值转换：\n',img_sobel_x_abs)
# 图像加权求值
img_sobel = cv2.addWeighted(img_sobel_x_abs,0.5,img_sobel_y_abs,0.5,0)
#laplace算子函数调用
img_laplace = cv2.Laplacian(gray,cv2.CV_64F,ksize=3)
#canny算子函数调用
img_canny = cv2.Canny(gray,120,280)
#prewitt算子卷积计算
pwitt_kernel_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
pwitt_kernel_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
img_pwitt_x = cv2.filter2D(gray,-1,pwitt_kernel_x)
img_pwitt_y = cv2.filter2D(gray,-1,pwitt_kernel_y)
print('卷积之后水平方向的数据：\n',img_pwitt_x)

img_pwitt = cv2.addWeighted(img_pwitt_x,0.5,img_pwitt_y,0.5,0)

plt.subplot(331),plt.imshow(gray,'gray'),plt.title('gray')
plt.subplot(332),plt.imshow(img_sobel_x,'gray'),plt.title('sobel_x')
plt.subplot(333),plt.imshow(img_sobel_y,'gray'),plt.title('sobel_y')
plt.subplot(334),plt.imshow(img_sobel,'gray'),plt.title('sobel')
plt.subplot(335),plt.imshow(img_laplace,'gray'),plt.title('laplace')
plt.subplot(336),plt.imshow(img_canny,'gray'),plt.title('Canny')
plt.subplot(337),plt.imshow(img_pwitt_x,'gray'),plt.title('prewitt_x')
plt.subplot(338),plt.imshow(img_pwitt_y,'gray'),plt.title('prewitt_y')
plt.subplot(339),plt.imshow(img_pwitt,'gray'),plt.title('prewitt')
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

'''
Sobel算子
Sobel算子函数原型如下：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
前四个是必须的参数：
第一个参数是需要处理的图像；
第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
其后是可选的参数：
dst是目标图像；
ksize是Sobel算子的大小，必须为1、3、5、7。
scale是缩放导数的比例常数，默认情况下没有伸缩系数；
delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''