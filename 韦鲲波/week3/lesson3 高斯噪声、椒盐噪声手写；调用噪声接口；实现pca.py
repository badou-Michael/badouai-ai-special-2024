import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from skimage import util
from sklearn.decomposition import PCA


'''
尝试一个高斯函数图像
切换mu和sigma的值看看效果
'''

def gauss_line():  # 尝试一个高斯函数图像，固定随机数种子
    a = 0
    func = []
    random.seed(0)
    plt.figure(figsize=(8, 200))  # 定义画布
    for sigma in range(0, 120, 20):  # 针对高斯分布的均值和方差进行不同数值搭配
        for mu in range(0, 120, 20):
            for i in range(100):
                q = random.gauss(mu, sigma)
                func.append(q)
            plt.title(f'q = {mu}, y = {sigma}')
            plt.plot(func)
            a += 1
            plt.subplot(36, 1, a)
    plt.tight_layout()
    plt.show()

#
def gauss(img, mu, sigma):  # 对图片实现高斯噪声
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    img_gauss = np.zeros((h, w), dtype=np.uint8)
    # print(h, w)
    for i in range(h):
        for j in range(w):
            q = random.gauss(mu, sigma)
            img[i, j] += q
            if img[i, j] > 255:
                img[i, j] = 255
            elif img[i, j] < 0:
                img[i, j] = 0
            img_gauss[i, j] = img[i, j]
    return img_gauss

def gauss_mult():  # lena高斯噪声的多个参数的对比效果
    img = cv2.imread('lenna.png')
    a = 1
    random.seed(0)
    # plt.imshow(gauss_img(img, 10, 10), cmap='gray')
    plt.figure(figsize=(8, 200))  # 定义画布
    for mu in range(0, 120, 20):  # 针对高斯分布的均值和方差进行不同数值搭配
        for sigma in range(0, 120, 20):
            plt.imshow(gauss(img, mu, sigma), cmap='gray')
            plt.title(f'mu = {mu}, sigma = {sigma}')
            plt.subplot(36, 1, a)
            a += 1
    plt.tight_layout()
    plt.show()

def salt_pepper(snr):
    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    count = snr * h * w
    for i in range(int(count)):
        q = random.random()
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        # print(q, x, y)
        if q <= 0.5:
            img[x, y] = 0
        elif q > 0.5:
            img[x, y] = 255
    plt.imshow(img, cmap='gray')
    plt.show()

def SP_mult():
    for i in np.arange(0, 1.1, 0.1):
        plt.title(f'snr = {i}')
        salt_pepper(i)

def guass(mean, sigma):
    img = plt.imread('lenna.png')
    guass = np.random.normal(mean, sigma, img.shape)
    img_guass = img + guass
    img_guass = np.clip(img_guass, 0, 255)
    plt.imshow(img_guass)
    plt.show()


'''
用skimage调用噪声生成函数
random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs)
mode就是选择用哪种噪声类型
gaussian 高斯噪声
localvar 高斯分布的加性噪声
poisson 泊松噪声
salt 盐噪声
pepper 椒噪声
s&p 椒盐噪声
speckle 均匀噪声
'''
def noise(img, noisetype):
    output = util.random_noise(img, mode=noisetype)
    return output


'''
PCA实现
1. 零均值化（中心化）
用每一个变量减去均值
即将零散的数据平移到原点附近，此时均值出来的线才能代表这些数据的真实趋势

2. 求协方差矩阵
方差：用来刻画随机变量x和数学期望均值的偏差的数学方法
在PCA中，方差越大，数据在坐标中越分散，说明该属性能够比较好的反映源数据
引入协方差矩阵的目的：
    降维后同一纬度的方差最大
    不同维度之间的相关性为0

因此协方差是用来度量【两个】随机变量【关系】的统计量
同一元素的协方差就表示该元素的方差，不同元素之间的协方差就表示他们的相关性
cov(X, Y) = ∑n,i=1 (Xi - X̄)(Yi - Ȳ) / (n - 1)
由此可见
Cov(X, X) = D(X), Cov(Y, Y) = D(Y)
协方差衡量了两个属性间的关系
当cov(X, Y) > 0时，表示X与Y正相关
当cov(X, Y) < 0时，表示X与Y负相关
当cov(X, Y) = 0时，表示X与Y不相关
'''

class CPCA:
    '''
    用类对象定义PCA的计算
    需要在类中定义很多类方法来进行各个环节的计算公式
    '''
    def __init__(self, X, K):  # 首先定义初始化内容，即调用类对象时直接进行计算
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值，即需要用多少特征值来进行计算PCA

        # 初始化一些后面需要用到的变量
        self.centerX = []  # 矩阵X的中心化，这个会通过用X来计算出来
        self.C = []  # 样本集的协方差矩阵
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的将为矩阵Z

        # 为后面类方法得出的结果赋值各自的变量
        self.centerX = self._centralized()  # 用来实现样本矩阵X中心化的类方法
        self.C = self._cov()  # 用来求样本矩阵X的协方差矩阵的类方法
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        '''
        样本矩阵X的中心化
        '''
        print('样本矩阵X', self.X)
        centerX = []  # 初始化变量

        mean = np.array([np.mean(attr) for attr in self.X.T])
        # 这个意思是，for遍历X的每一列（由X.T转置），通过np.mean计算每列的平均值，每个平均值加入列表

        print('样本矩阵X的均值', mean)
        centerX = self.X - mean  # 这个就是样本的中心化结果
        print('样本矩阵X的中心化', centerX)
        return centerX

    def _cov(self):
        '''
        求样本矩阵X中心化的协方差矩阵C
        '''
        ns = np.shape(self.X)[0]  # 样本矩阵X的样例总数，矩阵X的行数
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)  # 这里就是对于做了中心化的矩阵，去做协方差矩阵的公式
        print('样本矩阵X中心化的协方差矩阵C', C)
        return C

    def _U(self):
        '''
        求X的降维转换矩阵U，shape=(n, k)，n是X的特征维度总数，k是降维矩阵的特征维度。
        即求特征值和特征向量
        '''
        a, b = np.linalg.eig(self.C)  # 直接使用np.linalg.eig方法，即可得到中心化协方差矩阵C的特征值a，和特征向量b
        print('样本矩阵X中心化后的协方差矩阵C的特征值', a)
        print('样本矩阵X中心化后的协方差矩阵C的特征向量', b)
        ind = np.argsort(-1 * a)  # np.argsort是从小到大排序，-1的加入使特征值可以从大到小排序
        UT = [b[:, ind[i]] for i in range(self.K)]  # 构建K阶降维的降维转换矩阵U
        U = np.transpose(UT)  #
        print(f'{self.K}阶降维转换矩阵U', U)
        return U

    def _Z(self):
        '''
        按照Z=XU求降维矩阵Z，shape=(m, k)，n是样本总数，k是降维矩阵中特征维度总数
        '''
        Z = np.dot(self.X, self.U)
        print('X.shape', np.shape(self.X))
        print('U.shape', np.shape(self.U))
        print('Z.shape', np.shape(Z))
        print('样本矩阵X的降维矩阵Z', Z)
        return Z


'''简写一个PCA'''
class MPCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]

        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vecs = np.linalg.eig(self.covariance)

        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)

        # 降维矩阵
        self.components_ = eig_vecs[:, idx[:self.n_components]]

        # 对X进行降维
        return np.dot(X, self.components_)













if __name__ == '__main__':
    '''输出一组高斯线条，看看效果'''
    # gauss_line()

    '''lena高斯噪声的多个参数的对比效果'''
    # gauss_mult()

    '''实现椒盐噪声'''
    # salt_pepper(0.1)

    '''实现椒盐噪声图像多个值对比'''
    # SP_mult()

    '''高斯噪声的另一种写法'''
    # guass(0.1, 1)
    # guass(0.1, 2)
    # guass(0.1, 3)
    # guass(0.1, 4)


    '''用API实现噪声'''
    # img = cv2.imread('lenna.png')
    # q = noise(img, 'gaussian')
    # q2 = noise(img, 's&p')
    # q3 = noise(img, 'poisson')
    # cv2.imshow('img1', q)
    # cv2.waitKey(0)
    # cv2.imshow('img2', q2)
    # cv2.waitKey(0)
    # cv2.imshow('img3', q3)
    # cv2.waitKey(0)

    '''调用手写的CPCA输出样本X的降维矩阵'''
    # X = np.array(
    #     [
    #         [10, 15, 29],
    #         [15, 22, 3],
    #         [32, 20, 19],
    #         [28, 35, 2],
    #         [39, 24, 13],
    #         [70, 11, 9],
    #     ]
    # )
    #
    # K = np.shape(X)[1] - 1
    # print('样本矩阵X', X)
    # pca = CPCA(X, K)

    '''调用简写的接口输出PCA'''
    # X = np.array(
    #     [
    #         [10, 15, 29],
    #         [15, 22, 3],
    #         [32, 20, 19],
    #         [28, 35, 2],
    #         [39, 24, 13],
    #         [70, 11, 9],
    #     ]
    # )
    #
    # pca = MPCA(n_components=2)
    # print(pca.fit_transform(X))

    '''直接调用sklearn的PCA接口'''
    X = np.array(
        [
            [10, 15, 29],
            [15, 22, 3],
            [32, 20, 19],
            [28, 35, 2],
            [39, 24, 13],
            [70, 11, 9],
        ]
    )
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.fit_transform(X))


