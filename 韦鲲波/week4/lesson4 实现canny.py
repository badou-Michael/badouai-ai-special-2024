import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage import convolve

'''
canny需要由五部实现
0、灰度化：可做可不做
1、去噪声：应用高斯滤波来平滑图像，目的是去除噪声
2、梯度：通过边缘幅度的算法，找寻图像的梯度
3、非极大值抑制：应用非最大抑制技术来过滤掉非边缘像素，将模糊的边界变得清晰。该过程保留了每个像素点上梯度强度的极大值，过滤掉其他的值。
4、应用双阈值的方法来决定可能的（潜在的）边界；
5、利用滞后技术来跟踪边界。若某一像素位置和强边界相连的弱边界认为是边界，其他的弱边界则被删除。
'''


'''
========================================================================================================================
第零部分
对图片进行灰度化
'''

class Grayscale:  # 定义一个灰度化的类
    def __init__(self, img):
        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 目前只用cv2的灰度化方法



'''
========================================================================================================================
第一部分
关于去噪声，或者叫图像平滑
目前学的有以下几种图像平滑的滤波器

1、均值滤波
均值滤波是滤波器N*N的中心点，改为滤波器所有像素的平均值的方法。
即：滤波器像素值的和/滤波器像素数量【可以是3*3、5*5、7*7……】
可调用cv2中的blur方法实现，其中核大小通过元祖的方式传递，不能直接写一个值，所以可以写非正方形的滤波器
目前来看效果还是可以的

2、中值滤波
中值滤波是将滤波器N*N内所有像素按大小排序，取中值的方法。
例如：5*5的滤波器，有25个值，则取第13个值为滤波器的值
可调用cv2中的medianBlur方法实现，核大小可以直接用一个数实现
目前来看效果比较好

3、高斯滤波
高斯滤波是通过将每个滤波器对应位置的像素，与滤波器中的值进行乘积，最终在求和计算的方法，目前可知的有3*3、5*5的滤波器
例如：3*3的滤波器中，用原像素值p1*滤波器系数x1 + 原像素值p2*滤波器系数x2 + 原像素值p3*滤波器系数x3 + …… + 原像素值p9*滤波器系数x9
可调用cv2中的GaussianBlur方法实现，核大小需要用元祖传递，还需要传递一个方差的值
目前来看对图像的影响最小的
'''

class Filter:  # 定义一个滤波的类
    def __init__(self, img, fil, ksize, *args):
        self.args = args
        self.img = img  # 初始化导入图片
        self.fil = fil
        self.ksize = ksize  # 设置尺寸

        choice_fil = {
            'gauss': self.__gauss,
            'avg':self.__avg,
            'mid':self.__mid,
        }

        if self.fil in choice_fil:
            self.result = choice_fil[self.fil]()

    def __gauss(self):  # 高斯滤波
        result = cv2.GaussianBlur(self.img, self.ksize, self.args[0])
        print('调用gauss')
        return result


    def __avg(self):  # 均值滤波
        result = cv2.blur(self.img, self.ksize)
        print('调用avg')
        return result

    def __mid(self):  # 中值滤波
        result = cv2.medianBlur(self.img, self.ksize)
        print('调用mid')
        return result



class Filter_details:
    def __init__(self, img, sigma, dim):
        self.img = img
        self.sigma = sigma
        self.dim = dim
        self.dx, self.dy = img.shape
        # sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
        # dim = 5  # 高斯核尺寸
        self.Gaussian_filter = np.zeros([self.dim, self.dim])  # 存储高斯核，这是数组不是列表了
        self.tmp = [i-self.dim//2 for i in range(self.dim)]  # 生成一个序列
        self.gaussKernel = self.__kernel()
        self.filterResult = self.__filter()

    def __kernel(self):
        n1 = 1/(2*math.pi*self.sigma**2)  # 计算高斯核
        n2 = -1/(2*self.sigma**2)
        for i in range(self.dim):
            for j in range(self.dim):
                self.Gaussian_filter[i, j] = n1*math.exp(n2*(self.tmp[i]**2+self.tmp[j]**2))
        return self.Gaussian_filter / self.Gaussian_filter.sum()

    def __filter(self):
        img_new = np.zeros(self.img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
        self.tmp = self.dim//2
        img_pad = np.pad(self.img, ((self.tmp, self.tmp), (self.tmp, self.tmp)), 'constant')  # 边缘填补
        for i in range(self.dx):
            for j in range(self.dy):
                img_new[i, j] = np.sum(img_pad[i:i+self.dim, j:j+self.dim]*self.Gaussian_filter)
        return img_new


'''
========================================================================================================================
第二部分
寻找图像梯度

目前可用的边缘幅度算法有：sobel、prewitt、laplace、Canny算子

按照Sobel算子计算梯度幅值和方向，寻找图像的梯度。
先将卷积模板分别作用x和y方向，再计算梯度幅值和方向。

在用arctan2和np.sqrt后，需要用astype转换成float32或uint16、8，因为cv2imshow不支持float16类型
直接用cv2提供的Sobel方法就不用数据类型转换了
'''

class Gradient:
    def __init__(self, img):
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        # dx, dy = img.shape
        # img_tidu_x = np.zeros(img.shape)  # 存储梯度图像
        # img_tidu_y= np.zeros(img.shape)  # 存储梯度图像
        # self.img_tidu = np.zeros(img.shape)  # 存储梯度图像

        # img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
        # for i in range(dx):
        #     for j in range(dy):
        #         img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
        #         img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
        #         self.img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
        # img_tidu_x[img_tidu_x == 0] = 0.00000001
        # self.angle = img_tidu_y / img_tidu_x

        # 用cv2的卷积，用这个卷积后面输出角度和幅值时需要做astype(np.uint8)的数据类型转换，因为算完是float16的
        img_tidu_x = cv2.filter2D(img, -1, sobel_kernel_x)
        img_tidu_y = cv2.filter2D(img, -1, sobel_kernel_y)
        # self.img_tidu_x = convolve(img, sobel_kernel_x)
        # self.img_tidu_y = convolve(img, sobel_kernel_y)
        # self.img_tidu = np.sqrt(img_tidu_x**2 + img_tidu_y**2).astype(np.uint8)
        # self.angle = np.arctan2(img_tidu_y, img_tidu_x).astype(np.uint8)

        # 用cv2的Sobel算子直接卷，直接用cv2的Sobel方法就不用数据类型转换了
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
        self.img_tidu = np.sqrt(gx ** 2 + gy ** 2)
        self.angle = np.arctan2(gy, gx)






'''
========================================================================================================================
第三部分

非极大值抑制
'''

class NMS:
    def __init__(self, angle, mag):
        nms_img = np.zeros([mag.shape[0] + 2, mag.shape[1] + 2])

        # 弧度转角度，并只保留0到180度的角度
        angle = np.degrees(angle) % 180

        # 数组扩容
        mag_pad = np.pad(mag, ((1, 1), (1, 1)), 'constant')
        angle_pad = np.pad(angle, ((1, 1), (1, 1)), 'constant')
        high, weight = mag_pad.shape

        # NMS处理
        for i in range(1, high - 1):
            for j in range(1, weight - 1):
                p1 = p2 = 0
                if angle_pad[i, j] <= np.pi/8 or angle_pad[i, j] > 7 * np.pi/8:
                    p1, p2 = mag_pad[i - 1, j], mag_pad[i + 1, j]
                elif np.pi/8 < angle_pad[i, j] <= 3 * np.pi/8:
                    p1, p2 = mag_pad[i - 1, j - 1], mag_pad[i + 1, j + 1]
                elif np.pi/8 < 3 * angle_pad[i, j] <= 5 * np.pi/8:
                    p1, p2 = mag_pad[i, j - 1], mag_pad[i, j + 1]
                elif np.pi/8 < 5 * angle_pad[i, j] <= 7 * np.pi/8:
                    p1, p2 = mag_pad[i + 1, j - 1], mag_pad[i - 1, j + 1]

                # 判断是否局部最大值
                if mag_pad[i, j] > p1 and mag_pad[i, j] > p2:
                    nms_img[i, j] = mag_pad[i, j]

        self.nms_img = nms_img[1:high-1, 1:weight-1]





# def __init__(self, img):
#     self.dx, self.dy = img.shape
#     img_yizhi = np.zeros(img.shape)
#         for i in range(1, self.dx-1):
#             for j in range(1, self.dy-1):
#                 flag = True  # 在8邻域内是否要抹去做个标记
#                 temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
#                 if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
#                     num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
#                     num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
#                     if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
#                         flag = False
#                 elif angle[i, j] >= 1:
#                     num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
#                     num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
#                     if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
#                         flag = False
#                 elif angle[i, j] > 0:
#                     num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
#                     num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
#                     if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
#                         flag = False
#                 elif angle[i, j] < 0:
#                     num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
#                     num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
#                     if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
#                         flag = False
#                 if flag:
#                     img_yizhi[i, j] = img_tidu[i, j]



'''
========================================================================================================================
第四部分

双阈值和边缘连接

双阈值应用于强弱边缘

边缘连接应用于强弱边缘之间的
'''

class Duo_thre:
    def __init__(self, nms, low, mult, times):
        self.low = low
        self.mult = mult
        self.times = times
        self.__high = mult * low
        step1 = np.where(nms > low, nms, 0)
        step2 = np.where(step1 >= low * mult, 255, step1)
        self.step2 = step2

    def edgelink(self):
        # 扩展边缘
        duo_pad = np.pad(self.step2, ((1, 1), (1, 1)), 'constant')
        high, weight = duo_pad.shape
        source = np.where(duo_pad == 255, 255, 0)
        output = np.zeros([high, weight])

        # 执行输入的边缘处理次数
        while self.times == 0:
            # 对弱边缘进行处理
            for i in range(1, high - 1):
                for j in range(1, weight - 1):
                    if duo_pad[i, j] != 255:
                        continue
                    else:
                        kernel = duo_pad[i - 1:i + 2, j - 1:j + 2]
                        kernel_out = np.where(self.low <= kernel <= self.__high, 255, 0)
                        output += np.pad(kernel_out, ((i, high - 4 - i), (j, weight - 4 - j)), 'constant')
                        output = np.where(output > 255, 255, output)

            self.times -= 1

        output += source
        return output

'''
========================================================================================================================
'''








if __name__ == '__main__':
    # 灰度化
    img = cv2.imread('lenna.png')
    gray = Grayscale(img).gray_img

    # 调用Filter对象的gauss方法
    gauss = Filter(gray, 'gauss', (3, 3), 1000).result

    # 调用Gradient对象的angle属性和img_tidu属性
    angle = Gradient(gauss).angle
    grad_img = Gradient(gauss).img_tidu
    # cv2.imshow('111', angle)
    # cv2.imshow('1211', grad_img)

    # 非极大值抑制
    nms = NMS(angle, grad_img).nms_img
    # cv2.imshow('lenna', nms)

    # 双阈值&边缘链接
    output1 = Duo_thre(nms, 50, 1.5, 3).edgelink()
    output2 = Duo_thre(nms, 50, 2, 3).edgelink()
    output3 = Duo_thre(nms, 50, 2.5, 3).edgelink()
    output4 = Duo_thre(nms, 50, 3, 3).edgelink()
    output5 = Duo_thre(nms, 50, 3.5, 3).edgelink()

    output6 = Duo_thre(nms, 100, 1.5, 3).edgelink()
    output7 = Duo_thre(nms, 100, 2, 3).edgelink()
    output8 = Duo_thre(nms, 100, 2.5, 3).edgelink()
    output9 = Duo_thre(nms, 100, 3, 3).edgelink()
    output10 = Duo_thre(nms, 100, 3.5, 3).edgelink()


    resultr1 = np.hstack((output1, output2, output3, output4, output5))
    resultr2 = np.hstack((output6, output7, output8, output9, output10))

    result = np.vstack((resultr1, resultr2))

    cv2.imshow('lenna', result)
    # x = Gradient(gauss).img_tidu_x
    # y = Gradient(gauss).img_tidu_y
    # cv2.imshow('img', x)
    # cv2.imshow('img2', y)
    cv2.waitKey(0)




















