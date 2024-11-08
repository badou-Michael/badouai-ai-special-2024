# noinspection PyUnresolvedReferences
import cv2
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt

#灰度化
image=cv2.imread("lenna.png")
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#灰度图直方图
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
hist=cv2.calcHist([image_gray],[0],None,[256],[0,256])#histSize 要 [] 形式
plt.figure()#新建空白图像
plt.title("show gray hist")#图像名称
plt.xlabel("# Bins")#图像横轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.xlim([0,256])#设置x坐标轴范围
plt.plot(hist)
plt.show()

#灰度图像直方图均衡化
gray_equal=cv2.equalizeHist(image_gray)
ge_hist=cv2.calcHist([gray_equal],[0],None,[256],[0,256])
plt.figure()
plt.xlabel("#Bins")
plt.ylabel("# of Pixels")
plt.xlim([0,256])
plt.plot(ge_hist)
plt.show()


#彩色图像直方图
image_colored=cv2.imread("lenna.png")
#B,G,R=cv2.split(image_colored)#通道拆分
#chans=B,G,R
colors=("b","g","r")
plt.figure()#新建空白图像
plt.title("show colored hist")
plt.xlabel("#Bins")
plt.ylabel("# of Pixels")
plt.xlim([0, 256])

for chan,col in enumerate(colors):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    hist=cv2.calcHist([image_colored],[chan],None,[256],[0,256])#图像是chan。image_colored拆分了3个通道，chan分别对应 B,G,R三个通道的矩阵；channels:如果如图像是灰度图它的值就是[0]，如果是彩色图像的传入参数可以是[0][1][2]它们分别对应着BGR
    plt.plot(hist,color=col)
plt.show()

#彩色图像均衡化,需要分解通道 对每一个通道均衡化
#1 先分解三通道
image_colored=cv2.imread("lenna.png")
(b,g,r)=cv2.split(image_colored)#通道拆分
bh=cv2.equalizeHist(b)
gh=cv2.equalizeHist(g)
rh=cv2.equalizeHist(r)
#2 合并
result=cv2.merge((bh,gh,rh))
cv2.imshow("show colored_equal",result)
cv2.waitKey(0)
