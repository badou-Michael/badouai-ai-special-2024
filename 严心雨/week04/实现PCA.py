import numpy as np

class PCA(object): #object类是所有类的根类
    def __init__(self,X,K):#在定义__init__方法时，它的第一个参数应该是self,之后的参数用来初始化实例变量，调用构造方法时不需要传入self参数
      self.X=X #创建和初始化实例变量
      self.K=K
      self.centrX=[]#去中心化后的数组
      self.C=[]#协方差矩阵
      self.U=[]#求出特征向量的转换矩阵
      self.Z=[]#Z=X*U 降维后的新矩阵

      self.centrX=self.centralized()
      self.C=self.cov()
      self.U=self._U()#要加
      self.Z=self._Z()

    #定义实例方法时，它的第一个参数也应该是self,这会将当前实例与该方法绑定起来，这也说明该方法属于实例
    def centralized(self):#实例方法 去中心化 10*3
        #mean=np.array([np.mean(attr) for attr in self.X.T],dtype=object)#对X的列求均值=对X.T的行求均值 要加[]！
        print('X的转置矩阵\n',X.T)
        mean = np.array([np.mean(i) for i in self.X.T])
        print('每一行的样本均值\n', mean)
        centrX=self.X-mean # 样本集的中心化：变量 - 均值，平移后使得所有数据的中心是(0,0)
        print('中心化后的数据矩阵centrX\n', centrX)
        print('中心化后的数组矩阵centrX shape',centrX.shape)
        return centrX

    def cov(self):#实例方法 协方差矩阵
        ns=self.X.shape[0]#或者矩阵的行 即样本数量
        C=np.dot(self.centrX.T,self.centrX)/(ns-1)#ns-1 无偏估计 记住！ 注意点乘的矩阵位置 A*B 和 B*A结果不一致 公式是转置矩阵*原矩阵
        print('协方差矩阵\n',C)
        print('协方差矩阵shape',C.shape)#10*10
        return C

    def _U(self):#实例方法 求特征值和特征向量 获取转换矩阵
        a,b=np.linalg.eig(self.C)#特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        top=np.argsort(-1*a)
        print('特征向量\n',b)
        #将特征值按照从大到小的索引值顺序排序，选择其中最大的k个，然后将其对于的k个特征向量分别作为列向量组成转换矩阵
        UT=[b[:,top[i]] for i in range(self.K)]#b[:,x] 取b矩阵的所有行，和某列 用for循环，第一列+第二列
        U=np.transpose(UT)#transpose()函数的作用就是调换行列值的索引值，类似于矩阵的转置
        print('转换矩阵',U)
        print('转换矩阵shape',U.shape)
        return U

    def _Z(self):
        Z=np.dot(self.X,self.U)
        print('降维后新矩阵',Z)
        print('降维后新矩阵shape',Z.shape)
        return Z

if __name__== '__main__':##判断两者是否相等要用==！
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    #三维变二维
    K=X.shape[1]-1
    pca=PCA(X,K)#调用方法，不用写self参数
