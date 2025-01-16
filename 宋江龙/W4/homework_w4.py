# -*- coding:utf-8 -*-
# @Time:2024/9/25 19:17
# @author:sjlong

import cv2
import numpy as np
import random
from numpy import shape
import os
from skimage import util


class HomeworkWeek4Noise(object):

    def salt_pepper_noise(self, snr, image):
        '''椒盐噪声：
        1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
        2.计算总像素数目SP，得到要加噪的像素数目 NP = SP * SNR，SP = 图片的高*宽
        3.随机获取要加噪的每个像素位置P（i, j）
        4.指定像素值为255或者0，取的可能性都是0.5
        5.重复3, 4两个步骤完成所有NP个像素的加噪
        '''
        if 0 < snr < 1:
            np = int(snr * image.shape[0] * image.shape[1])
            for i in range(np):
                h = random.randint(0, image.shape[0] - 1)  # -1是为了防止越界，如果边缘不处理，则是-2
                w = random.randint(0, image.shape[1] - 1)  # 同上
                if random.random() <= 0.5:  # random.random()随机生成0-1之间的数据
                    image[h, w] = 0
                else:
                    image[h, w] = 255
            return image
        else:
            print('信噪比需在0-1之间')

    def gaussian_noise(self, snr, image, mean, sigma):
        '''高斯噪声：
        a. 指定信噪比 0 < SNR（信号和噪声所占比例） < 1，计算总像素数目SP，得到要加噪的像素数目 NP = SP * SNR，SP = 图片的高*宽
        b. 在random.gauss中输入参数sigma 和 mean，生成高斯随机数
        d. 在原有像素灰度值上加上生成的高斯随机数
        e. 重新将像素值放缩在[0 ~ 255]之间，小于0的，强制转成0，大于255的，强制转成255
        f. 循环所有像素
        g. 输出图像
        '''
        if 0 < snr < 1:
            np = int(snr * image.shape[0] * image.shape[1])
            for i in range(np):
                h = random.randint(0, image.shape[0] - 1)
                w = random.randint(0, image.shape[1] - 1)
                image[h, w] += random.gauss(mean, sigma)
                if image[h, w] < 0:
                    image[h, w] = 0
                elif image[h, w] > 255:
                    image[h, w] = 255
            return image
        else:
            print('信噪比需在0-1之间')

    def invoke_noise_interface(self, image, mode, **kwargs):
        '''噪声接口调用
        mode： 可选择，str型，表示要添加的噪声类型
            gaussian：高斯噪声
            localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
            poisson：泊松噪声
            salt：盐噪声，随机将像素值变成1
            pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
            s&p：椒盐噪声
            speckle：均匀噪声（均值mean方差variance），out=image+n*image
        seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
        clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果是False，则输出矩阵的值可能会超出[-1,1]
        mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
        var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
        local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
        amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
        salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
        --------
        返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
        '''
        result = util.random_noise(image, mode=mode, **kwargs)
        return result


    def file_path(self):
        '''获取图片路径'''
        base_path = os.path.abspath(os.path.dirname(__file__))
        data_path = os.path.join(base_path, 'data')
        file_path = os.path.join(data_path, 'lenna.png')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        return data_path, file_path


class HomeworkWeek4PCA(object):
    '''实现PCA：传入矩阵matrix和需要实现的维度component参数'''
    def __init__(self, matrix, component):
        self.matrix = matrix
        self.component = component
        # self.centralized, self.covariance, self.eigenVectorMatrix, self.down_matrix = [],[],[],[]
        self.centralized = self._centralized()
        self.covariance = self._covariance()
        self.eigenVectorMatrix = self._eigenVectorMatrix()
        self.down_matrix = self._down_matrix()

    def _centralized(self):
        '''对原始数据零均值化（中心化）:centralized'''
        matrix_mean =self.matrix - self.matrix.mean(axis=0)
        return matrix_mean

    def _covariance(self):
        '''求原始矩阵的协方差矩阵covariance'''
        n = self.centralized.shape[0]
        covariance = np.dot(self.centralized.T, self.centralized) / n
        return covariance

    def _eigenVectorMatrix(self):
        '''根据协方差矩阵求特征值和特征向量'''
        eig_value, eig_vector = np.linalg.eig(self.covariance)  # 特征值赋值给eig_value，对应特征向量赋值给eig_vector
        # 给出特征值降序的索引序列
        idh = np.argsort(-1 * eig_value)
        # 求特征向量矩阵
        eigenVectorMatrix = eig_vector[:, idh[: self.component]]
        return eigenVectorMatrix

    def _down_matrix(self):
        '''求最终的降维矩阵：原矩阵matrix * 特征向量矩阵eigenVectorMatrix'''
        down_matrix = np.dot(self.matrix, self.eigenVectorMatrix)
        return down_matrix


if __name__=='__main__':
    homework_w4_noise = HomeworkWeek4Noise()
    image=cv2.imread(homework_w4_noise.file_path()[1], 0)  # 0表示灰度，1表示彩色
    salt_pepper_noise_image = homework_w4_noise.salt_pepper_noise(0.2, image)
    gaussian_noise_image = homework_w4_noise.gaussian_noise(0.8, image, 8, 16)
    invoke_interface = homework_w4_noise.invoke_noise_interface(image, mode='s&p', amount=0.1)
    # cv2.imwrite(homework_w4_noise.file_path()[0] + '\lenna_PepperandSalt.png', salt_pepper_noise_image)  # 在指定文件夹中生成图片
    # cv2.imwrite(homework_w4_noise.file_path()[0] + '\gaussian_noise.png', gaussian_noise_image)
    cv2.imshow('salt-pepper-noise picture', salt_pepper_noise_image)
    cv2.imshow('gaussian_noise picture', gaussian_noise_image)
    cv2.imshow('invoke_interface picture', invoke_interface)
    cv2.waitKey(0)
    X = np.array([[-1, 2, 66, -1],
                  [-2, 6, 58, -1],
                  [-3, 8, 45, -2],
                  [1, 9, 36, 1],
                  [2, 10, 62, 1],
                  [3, 5, 83, 2]])  # 导入数据，维度为4
    homework_w4_pca = HomeworkWeek4PCA(X, 2)
    down_matrix = homework_w4_pca._down_matrix()
    print(down_matrix)
