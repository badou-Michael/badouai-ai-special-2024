
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class CANNY(object):

    def __init__(self, gray, sigma, dim):
        self.gray = gray
        self.sigma = sigma
        self.dim = dim
        self.gaussianImg = []
        self.gradientImg = []
        self.angle = []
        self.inhibitionImg = []
        self.thresholdImg = []

        self.gaussianImg = self._gauss()
        self.gradientImg, self.angle = self._gradientImg()
        self.inhibitionImg = self._inhibitionImg()
        self.thresholdImg = self._thresholdImg()

    def _gauss(self):
        Gaussian_filter = np.zeros([self.dim, self.dim])
        temp = [i-self.dim//2 for i in range(self.dim)]
        for i in range(self.dim):
            for j in range(self.dim):
                Gaussian_filter[i, j] = math.exp(-(temp[i]**2+temp[j]**2)/(2*self.sigma**2))/(2*math.pi*self.sigma**2)
        Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
        print("高斯卷积核:",Gaussian_filter)
        dx, dy = self.gray.shape
        gaussianImg = np.zeros([dx,dy])
        img_padding = np.pad(self.gray, ((2, 2), (2, 2)), 'constant')
        for i in  range(dx):
            for j in range(dy):
                gaussianImg[i, j] = np.sum(img_padding[i:i+self.dim, j:j+self.dim]*Gaussian_filter)
        plt.figure(1)
        plt.imshow(gaussianImg.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return gaussianImg

    def _gradientImg(self):
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        dx,dy = self.gaussianImg.shape
        sobel_gradient_x = np.zeros([dx,dy])
        sobel_gradient_y = np.zeros([dx,dy])
        gradientImg = np.zeros([dx,dy])
        img_padding = np.pad(self.gaussianImg, ((1, 1),(1, 1)), 'constant')
        for i in range(dx):
            for j in range(dy):
                sobel_gradient_x[i,j] = np.sum(img_padding[i:i+3, j:j+3]*sobel_kernel_x)
                sobel_gradient_y[i,j] = np.sum(img_padding[i:i+3, j:j+3]*sobel_kernel_y)
                gradientImg[i, j] = np.sqrt(sobel_gradient_x[i,j]**2 + sobel_gradient_y[i,j]**2)
        sobel_gradient_x[sobel_gradient_x==0] = 0.00000001
        angle = sobel_gradient_y/sobel_gradient_x
        plt.figure(2)
        plt.imshow(gradientImg.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return gradientImg, angle

    def _inhibitionImg(self):
        dx, dy = self.gradientImg.shape
        img_inhibition = np.zeros([dx, dy])
        for i in range(1,dx-1):
            for j in range(1,dy-1):
                temp = self.gradientImg[i-1:i+2, j-1:j+2]
                flag = True
                if self.angle[i,j] <= -1:
                    num_1 = (temp[0, 1]-temp[0, 0])/self.angle[i,j] + temp[0,1]
                    num_2 = (temp[2, 1]-temp[2, 2])/self.angle[i,j] + temp[2,1]
                    if not (self.gradientImg[i, j] > num_1 and self.gradientImg[i,j] > num_2):
                        flag = False
                if self.angle[i,j] >= 1:
                    num_1 = (temp[0, 2]-temp[0, 1])/self.angle[i, j] + temp[0,1]
                    num_2 = (temp[2, 0]-temp[2, 1])/self.angle[i, j] + temp[2, 1]
                    if not (self.gradientImg[i,j] > num_1 and self.gradientImg[i,j] > num_2):
                        flag = False
                if self.angle[i, j] > 0:
                    num_1 = (temp[0, 2] - temp[1, 2]) * self.angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) * self.angle[i, j] + temp[1, 0]
                    if not (self.gradientImg[i, j] > num_1 and self.gradientImg[i, j] > num_2):
                        flag = False
                if self.angle[i,j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) * self.angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) * self.angle[i, j] + temp[1, 2]
                    if not (self.gradientImg[i,j]> num_1 and self.gradientImg[i,j] > num_2):
                        flag = False
                if flag:
                    img_inhibition[i, j] = self.gradientImg[i, j]
        plt.figure(3)
        plt.imshow(img_inhibition.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return img_inhibition


    def _thresholdImg(self):
        lower_boundary = self.gradientImg.mean() * 0.5
        high_boundary = lower_boundary * 3
        dx, dy = self.inhibitionImg.shape
        thresholdImg = self.inhibitionImg

        for i in range(1,dx-1):
            for j in range(1,dy-1):
                if thresholdImg[i,j] >= high_boundary:
                    thresholdImg[i,j] = 255
                elif thresholdImg[i,j] <= lower_boundary:
                    thresholdImg[i,j] = 0
                else:
                    temp = thresholdImg[i-1:i+2,j-1:j+1]
                    for x in range(temp.shape[0]):
                        for y in range(temp.shape[1]):
                            if temp[x,y] > high_boundary:
                                thresholdImg[i,j] = 255

        plt.figure(4)
        plt.imshow(thresholdImg.astype(np.uint8), cmap='gray')
        plt.axis('off')
        return thresholdImg



if __name__=='__main__':
    img = cv2.imread("C:/Users/bq-twenty-one/Desktop/123.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sigma = 0.5
    dim = 5
    CANNY(gray,sigma,dim)
    plt.show()
