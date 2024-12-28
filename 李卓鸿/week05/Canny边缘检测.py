import numpy
import math
import cv2
from matplotlib import pyplot

numpy.set_printoptions(suppress=True)

class My_canny(object):
    def __init__(self,gray_img,k_size:int,sigma,min_threshold:int,max_threshold:int):
        self.gray_img = gray_img
        self.k_size = k_size
        self.sigma = sigma

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.radius = int( self.k_size // 2 )

        self.kernel = self.GaussianKernel()

        self.sobel_img,self.sobel_angel = self.Sobel()

        self.nms_img = self.NMS_Methods()

        self.DoubleThreshold_img = self.double_threshold()

    def GaussianKernel(self): #高斯滤波核
        radius = int( self.k_size // 2 )
        kernel = numpy.zeros((self.k_size,self.k_size))
        com = 1 / ( 2 * numpy.pi * ( self.sigma ** 2 ) )
        s = 0
        for x in range(self.k_size):
            for y in range(self.k_size):
                kernel[x,y] = com * (math.exp(-1 * ( (( x - radius ) ** 2 ) + (( y - radius ) ** 2) ) / (2 * (self.sigma ** 2)) ))
                s += kernel[x,y]

        kernel = kernel / s

        return kernel

    def GaussianFilter(self): #高斯滤波
        h,w = self.gray_img.shape
        new_img = numpy.zeros((h + 2 * self.radius , w + 2 * self.radius))
        GF_img = numpy.zeros((h,w))

        for i in range(self.radius , h + self.radius):
            for j in range(self.radius , w + self.radius):
                new_img[i,j] = self.gray_img[ i - self.radius , j - self.radius ]

        for a in range(h):
            for b in range(w):
                part_img = new_img[a:(a + self.k_size) , b:(b + self.k_size)]
                GF_img[a,b] = numpy.sum(part_img * self.kernel)  # 高斯滤波核加权平均

        return GF_img.astype(int)

    def Sobel(self): # sobel边缘检测
        h,w = self.gray_img.shape
        s_x = numpy.array( [[-1,0,1],[-2,0,2],[-1,0,1]] )
        s_y = numpy.array( [[1,2,1],[0,0,0],[-1,-2,-1]] )

        G_x = numpy.zeros((h,w))
        G_y = numpy.zeros((h,w))
        G_xy = numpy.zeros((h,w))

        new_img = numpy.zeros((h + 2 , w + 2))

        for i in range(1 , h + 1):  #边缘填充
            for j in range(1 , w + 1):
                new_img[i,j] = self.gray_img[ i - 1 , j - 1 ]

        for a in range(h):
            for b in range(w):
                part_img = new_img[a:(a + 3) , b:(b + 3)]
                G_x[a,b] = numpy.sum(part_img * s_x)  # 横向卷积
                G_y[a,b] = numpy.sum(part_img * s_y)  # 纵向卷积
                G_xy[a,b] = numpy.sqrt(G_x[a,b] ** 2 + G_y[a,b] ** 2)  # 计算梯度大小

        angel = numpy.atan2(G_y,G_x)
        return G_xy,angel

    def NMS_Methods(self):  # 非极大值抑制
        h,w = self.gray_img.shape
        nms_img = numpy.zeros((h,w))

        p = math.pi
        first = (1/4) * p
        second = (3/4) * p
        third = (-3/4) * p
        forth = (-1/4) * p

        for i in range(1,h-1):
            for j in range(1,w-1):
                angel_xy = self.sobel_angel[i,j]
                temp = self.sobel_img[ (i - 1) : (i + 2) ,  (j - 1) : (j + 2) ]
                temp_up = float(0)
                temp_down = float(0)
                
                if (0 <= angel_xy) and (angel_xy < first) : # 0-45
                    k = angel_xy / first
                    temp_up = k * temp[ 0 , 2 ] + ( 1 - k ) * temp[ 1 , 2 ]
                    temp_down = k * temp[ 2 , 0 ] + ( 1 - k ) * temp[ 1 , 0 ]
                elif (first <= angel_xy) and ( angel_xy < (1/2 * p) ) : # 45-90
                    k = angel_xy / (1/2 * p)
                    temp_up = k * temp[ 0 , 1 ] + ( 1 - k ) * temp[ 0 , 2 ]
                    temp_down = k * temp[ 2 , 1 ] + ( 1 - k ) * temp[ 2 , 0 ]
                elif ((1/2 * p) <= angel_xy) and ( angel_xy < second ) : # 90-135
                    k = angel_xy / second
                    temp_up = k * temp[ 0 , 0 ] + ( 1 - k ) * temp[ 0 , 1 ]
                    temp_down = k * temp[ 2 , 2 ] + ( 1 - k ) * temp[ 2 , 1 ]
                elif (second <= angel_xy) and (angel_xy < p):  # 135-180
                    k = angel_xy / p
                    temp_up = k * temp[ 1 , 0] + (1 - k) * temp[ 0, 0]
                    temp_down = k * temp[ 1 , 2] + (1 - k) * temp[ 2, 2]
                elif angel_xy < third :  # 180-225
                    k = angel_xy / (-1 * p)
                    temp_up = k * temp[ 1 , 0] + (1 - k) * temp[ 2, 0]
                    temp_down = k * temp[ 1 , 2] + (1 - k) * temp[ 0, 2]
                elif (third <= angel_xy) and (angel_xy < (-1/2 * p)) :  # 225-270
                    k = angel_xy / third
                    temp_up = k * temp[ 2 , 0] + (1 - k) * temp[ 2, 1 ]
                    temp_down = k * temp[ 0 , 2] + (1 - k) * temp[ 0, 1 ]
                elif ( (-1/2 * p) <= angel_xy) and (angel_xy < forth) :  # 270-315
                    k = angel_xy / (-1/2 * p)
                    temp_up = k * temp[ 2 , 1 ] + (1 - k) * temp[ 2, 2 ]
                    temp_down = k * temp[ 0 , 1 ] + (1 - k) * temp[ 0, 2 ]
                else: # 315-360
                    k = angel_xy / forth
                    temp_up = k * temp[ 2 , 2 ] + (1 - k) * temp[ 1 , 2 ]
                    temp_down = k * temp[ 0 , 0 ] + (1 - k) * temp[ 0, 1 ]

                sobel_value = self.sobel_img[i,j] #当前像素值
                G_max = max(temp_up,temp_down,sobel_value) #比较当前像素值和梯度方向像素值
                if(sobel_value < G_max): # 当前像素值小于梯度方向像素值，置0
                    nms_img[i,j] = 0
                else: # 当前像素值大于梯度方向像素值，保留
                    nms_img[i,j] = sobel_value

        return nms_img.astype(int)

    def double_threshold(self):
        h,w = self.gray_img.shape
        DoubleThreshold_Img = numpy.zeros((h,w))
        for i in range(1,h-1):
            for j in range(1,w-1):
                value = self.nms_img[i,j]
                temp = self.nms_img[ (i - 1) : (i + 2) , (j - 1) : (j + 2) ]
                if value < self.min_threshold :
                    DoubleThreshold_Img[i,j] = 0
                elif value > self.max_threshold :
                    DoubleThreshold_Img[i,j] = value
                else:
                    max_value = numpy.max(temp)
                    if max_value >= self.max_threshold :
                        DoubleThreshold_Img[i,j] = value
                    else:
                        DoubleThreshold_Img[i,j] = 0

        return DoubleThreshold_Img.astype(int)

if __name__ == '__main__':
    img = cv2.imread('lenna.png',1)  #灰度化
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pyplot.subplot(231)
    pyplot.title('Source Img')
    pyplot.imshow(gray_img, cmap='gray')

    size,sigma = input('请输入高斯滤波器核的size，sigma，用空格隔开：').split()
    max_value,min_value = input('请输入Canny边缘检测的阈值上界，阈值下界，用空格隔开：').split()

    test = My_canny(gray_img,int(size),float(sigma),int(min_value),int(max_value))
    GF_img = test.GaussianFilter()

    pyplot.subplot(232)
    pyplot.title('GaussianFilter Img')
    pyplot.imshow(GF_img, cmap='gray')

    Sobel_img = test.sobel_img
    Sobel_angel = test.sobel_angel
    pyplot.subplot(233)
    pyplot.title('Sobel Img')
    pyplot.imshow(Sobel_img, cmap='gray')

    mns_img = test.nms_img
    pyplot.subplot(234)
    pyplot.title('NMS Img')
    pyplot.imshow(mns_img, cmap='gray')

    DoubleThreshold_img = test.DoubleThreshold_img
    pyplot.subplot(235)
    pyplot.title('DoubleThreshold Img')
    pyplot.imshow(DoubleThreshold_img, cmap='gray')

    pyplot.show()

