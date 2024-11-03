import cv2
import numpy as np
from numpy import shape
import random
from PIL import Image
from skimage import util

def gaussiannoise(souce, means, sigma, percentage):
    noiseimg = souce
    noisenum = int(percentage*souce.shape[0]*souce.shape[1])
    for i in range(noisenum):
        randx = random.randint(0, souce.shape[0]-1)
        randy = random.randint(0, souce.shape[1]-1)
        noiseimg[randx, randy] = noiseimg[randx, randy] + random.gauss(means, sigma)
        if noiseimg[randy, randy] < 0:
           noiseimg[randx, randy] = 0
        elif noiseimg[randx, randy] > 255:
             noiseimg[randx, randy] = 255
    return noiseimg

img = cv2.imread('lenna.png', 0)
img1 = gaussiannoise(img, 3, 3, 0.7)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#创建高斯图
#cv2.imwrite("lennagaussnoise.png", img1)
cv2.imshow("原图 :", img2)
cv2.imshow("高斯图:", img1)
cv2.waitKey(0)

def jzfun(src,percentage):
    noiseimg = src
    noisenum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(noisenum):
        randx = random.randint(0, src.shape[0]-1)
        randy = random.randint(0, src.shape[1]-1)
    if random.random() <= 0.5:
       noiseimg[randx, randy] = 0
    else:
       noiseimg[randx, randy] = 255
    return noiseimg
img=cv2.imread('lenna.png', 0)
img1=jzfun(img, 0.6)
#创建jz图
# cv2.imwrite("jznoise.png", img1)
cv2.imshow('jz图：', img1)
cv2.waitKey(0)

#噪声接口调用

#def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
#功能：为浮点型图片添加各种随机噪声
#参数：
#image：输入图片（将会被转换成浮点型），ndarray型
#mode： 可选择，str型，表示要添加的噪声类型
	#gaussian：高斯噪声
	#localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
	#poisson：泊松噪声
	#salt：盐噪声，随机将像素值变成1
	#pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
	#s&p：椒盐噪声
	#speckle：均匀噪声（均值mean方差variance），out=image+n*image
#seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
#clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果是False，则输出矩阵的值可能会超出[-1,1]
#mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
#var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
#local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
#amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
#salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
#返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数

img = cv2.imread("lenna.png")
gsnoiseimg = util.random_noise(img, mode='salt')
cv2.imshow('lennagaussin', gsnoiseimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


import numpy as np
from sklearn import PCA

class pca(object):

      def __init__(self,x,k):
          self.X = x #样本矩阵x
          self.K = k #样本矩阵k,是矩阵X的降维矩阵
          self.centerX = [] #矩阵X的中心化,设定为空
          self.C = [] #样本矩阵X的协方差矩阵C,设定为空
          self.U = [] #样本矩阵X的降维转换矩阵,设定为空
          self.Z = [] #样本矩阵的降维矩阵Z,设定为空

          self.centerX = self._ceteralized()#中心化
          self.c = self._cov() #协方差矩阵
          self.u = self._U() #降维矩阵u
          self.z = self._Z() #Z=XU


      def _ceteralized(self):#样本矩阵的中心化
          print("样本的矩阵x\n",self.X)
          centerX = []
          mean = np.array([np.mean(avg) for avg in self.X.T])
          print('样本集的特征均值：\n', mean)
          centerX = self.X - mean #样本中心化
          print('样本矩阵X的中心化centerx：\n',centerX)
          return centerX

      def _cov(self): #样本矩阵X的协方差矩阵c
          num = np.shape(self.centerX)[0] #样本集的样例总数
          C = np.dot(self.centerx.T, self.centerx) / num-1
          print('样本矩阵x的协方差矩阵c：\n', C)
          return  C

      def _U(self): #矩阵x的降维矩阵u，shape=(n,k),n是x的特征维度总数，k是降维矩阵的特征维度
          A, B = np.linalg(self.c) #特征值给A，对应的向量赋值给B
          print('样本矩阵x的协方差矩阵C的特征值A:\n')
          print('样本矩阵x的协方差矩阵C的特征值向量B:\n')
          ind = np.argsort(-1*A)#默认升序，-1降序
          UT = [B[:, ind[i]] for i in range(self.k)]
          U = np.transpose(UT)
          print('%d样本矩阵x的降维转换矩阵u：\n'%sel.k,U)
          return U

      def _Z(self): #降维矩阵Z，Z=XU,shape = (m,k) n是样本总数，k是佳能为矩阵中特征维度总数
          Z = np.dot(self.X , self.U)
          print('X SHAPE:',np.shape(self.X))
          print('U SHAPE:', np.shape(self.U))
          print('Z SHAPE:', np.shape(Z))
          print('样本矩阵X的降维矩阵Z:\n', Z)
          return  Z

if  __name__ == '__main__': #10样本3特征的样本集，行为样例，列为特征维度
        X = np.array([[15,15,4],
                  [4,8,9],
                  [15,7,28],
                  [6.12,36],
                  [5,70,63],
                  [20,10,26],
                  [24,5,13],
                  [24,6,17],
                  [10,7,11],
                  [18,8,13]])

        K = np.shape(X)[1] -1
        print('样本集10行3列，10个样本，每个样本3特征：\n', X)
        PCARESULT =pca(X,K)

###简化PCA###
#class PCA():
#     def __init__(self,n_components):
#         self.n_components = n_components
#
#            def fit_transform(self,x):
#         self.n_features = x.shape[1]
#         x = x - x.means(axis=0)
#         self.covariance = np.dot(x.t, x)/x.shape[0]
#         eig_vals, eig_vectors= np.linalg.eig(self.covariance)
#         idx =np.argsort(-eig_vals)
#        self.components = eig_vectors[:, idx[:self.n_components]]
#         return np.dot(x,self.components_)

###调用
#pca = PCA(n_components= 2)
#x = np.array([[15,15,4],[4,8,9],[15,7,28],[5,70,63],[20,10,26],[24,5,13],[24,6,17],[10,7,11],[18,8,13]])
#newx = pca.fit_transform(x)
#print(newx)



