import numpy as np


# 实现将样本从给定维度降维(k) 目的就是为了降维
class PCA():
    def __init__(self,x,k):
        # 样本
        self.x=x
        # 降维的维度
        self.k=k

        # 零均值化 优化结果
        self.centX=[]

        # 协方差矩阵 降维后相关性为0 因此需要求出协方差矩阵
        self.cm=[]

        # 特征值矩阵
        self.em= []
        # 结果

        self.result=[]

        self.centX=self.centralised()
        self.cm=self.cov() # 协方差矩阵
        self.em=self._em() # 特征矩阵
        self.result=self._result() # 结果矩阵

    # 中心化处理 (3,10)
    def centralised(self):
        # 每个特征即每一行对应的均值
        mean =np.array([np.mean(i) for i in self.x.T])
        print("每一行样本均值\n",mean)
        # 利用广播机制进行相减 np对mean进行扩展
        centX=self.x-mean
        print("中心化后的数据矩阵\n",centX)
        return centX

    # 求协方差矩阵shape=(3*3)
    def cov(self):
        num=np.shape(self.x)[0]
        covm=np.dot(self.x.T,self.x)/(num-1)
        print("协方差矩阵为\n",covm)
        return covm

    # 求特征矩阵 shape=(n,k) /(3*2) n 是特征总数
    def _em(self):
        # 先求特征值和特征向量
        a,b = np.linalg.eig(self.cov())
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 对a进行降序排序 argsort返回降序后特征值矩阵的索引
        i = np.argsort(-1*a)
        # b[:,x] 取b矩阵的所有行，和某列
        # 取较大的前两列
        emT = [b[:, i[j]] for j in range(self.k)]
        em = np.transpose(emT)
        print("10*2的特征矩阵为\n",em)
        return em

    def _result(self):
        result=np.dot(self.x,self.em)
        print("协方差的shape", np.shape(self.cm))
        print("中心化的shape",np.shape(self.centX))
        print("特征矩阵的shape", np.shape(self.em))
        print("降维后矩阵的shape", np.shape(result))
        print("降维后的矩阵\n",result)
        return result

if __name__=='__main__':
    # 10*3矩阵 3个维度
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    # 三维变二维
    k = np.shape(X)[1]-1     # 取列数减一
    print("样本集(10*3)\n",X)
    pca = PCA(X,k)
