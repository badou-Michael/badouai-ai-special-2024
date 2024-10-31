from skimage import util # 导入噪声模块
import matplotlib.pyplot as plt # 绘图模块
import cv2

#image=plt.imread('lenna.png') #读出来的格式直接就是ndarray-数组的格式
image=cv2.imread('lenna.png')
#plt.imshow(image)
#plt.show()
# 高斯噪音
image_noise_gaussi = util.random_noise(image,'gaussian',clip=True,var=0.1) #var 方差 方差越大，噪声越明显
cv2.imshow('show image_noise_gaussi',image_noise_gaussi)
cv2.waitKey(0)
cv2.destroyWindow('show image_noise_gaussi')#随意按任意键就会关闭窗口

#椒盐噪音
image_noise_jiaoyan1=util.random_noise(image,'s&p',clip=True,amount=0.5)
cv2.imshow('show image_noise_jiaoyan1',image_noise_jiaoyan1)
cv2.waitKey(0)

#用plt读的图放到噪声接口里，出的结果是蓝色的。因为opencv读图是bgr顺序，plt是rgb的，用同样的方式加噪声，再show出来，结果不一样
