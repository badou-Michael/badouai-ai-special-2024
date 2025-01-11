import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, inputnodes, midnodes, outputnodes, learning_rate):
        #  初始化神经网络
        self.innodes = inputnodes
        self.midnodes = midnodes
        self.outnodes = outputnodes

        # 初始化学习率
        self.lrate = learning_rate

        # 初始化权重
        tmp1 = np.random.rand(self.midnodes, self.innodes)  # 把innodes放后面是限制与它做点乘的数组高度，midnodes放前面代表输出多高
        # w_in2mid是个mid行in列矩阵
        self.w_in2mid = np.where(tmp1 == (tmp1 - np.mean(tmp1)) / (np.max(tmp1) - np.min(tmp1)), tmp1, (tmp1 - np.mean(tmp1)) / (np.max(tmp1) - np.min(tmp1)))
        # self.w_in2mid = tmp1 - 0.5
        # print('self.w_in2mid.shape', self.w_in2mid)

        tmp2 = np.random.rand(self.outnodes, self.midnodes)
        # w_mid2out是个out行mid列矩阵
        self.w_mid2out = np.where(tmp2 == (tmp2 - np.mean(tmp2)) / (np.max(tmp2) - np.min(tmp2)), tmp2, (tmp2 - np.mean(tmp2)) / (np.max(tmp2) - np.min(tmp2)))
        # self.w_mid2out = tmp2 - 0.5
        # print('self.w_mid2out.shape', self.w_mid2out)

        # 定义一个激活函数，sigmoid
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self, train_list, label_list):
        '''
        ========================================
        正向过程
        '''
        Zmid = np.dot(self.w_in2mid, train_list)  # input → mid  w_in2mid是mid行ni列，train_list是in行1列的数组，结果是mid行1列
        # print('Zmid.shape', Zmid.shape)

        Amid = self.activation_function(Zmid)  # mid → sigmoid  midn_outp也是mid行1列
        # print('Amid.shape', Amid.shape)

        Zout = np.dot(self.w_mid2out, Amid)  # mid → output  w_mid2out是out行mid列，此时输入的是mid行1列数组，输出的是out行1列的数组
        # print('Zout.shape', Zout.shape)

        Aout = self.activation_function(Zout)  # output → sigmoid  outn_outp也是out行1列
        # print('Aout.shape', Aout.shape)

        '''
        ========================================
        逆向过程
        首先是输出层权重
        偏导影响的顺序是
        
        ∂Etotal     ∂Etotal     ∂Aout     ∂Zout
        ———————  =  ———————  *  —————  *  —————
         ∂Wout       ∂Aout      ∂Zout     ∂Wout
        
        A代表输出数据，out代表输出层，Z代表输入数据，W代表权重
        '''

        # print('label_list.shape', label_list.shape)
        # 计算w_mid2out更新所需要的参数
        Et_ao = -(label_list - Aout)  # Etotal对输出层输出Aout的求导，结果是out行1列
        # print('Et_ao.shape', Et_ao.shape)

        ao_zo = Aout * (1 - Aout)  # Aout对输出层输入Zout的求导，结果是out行1列
        # print('ao_zo.shape', ao_zo.shape)

        Et_zo = (Et_ao * ao_zo).reshape(-1, 1)  # 上面两个想成，结果还是out行1列
        # print('Et_zo.shape', Et_zo.shape)

        zo_wo = Amid.reshape(1, -1)  # Zout对进入输出层的权重W的求导，结果实际就是Amid，mid行1列
        # print('zo_wo.shape', zo_wo.shape)

        Et_wo = np.dot(Et_zo, zo_wo)  # Etotal对Wout的求导，为了在下一步求出来的shape与mid2out的out行mid列相同，因此点乘的顺序要严格
        # print('Et_wo.shape', Et_wo.shape)

        '''
        ========================================
        其次是中间层权重
        偏导影响的顺序是

        ∂Etotal     ∂Etotal     ∂Amid     ∂Zmid
        ———————  =  ———————  *  —————  *  —————
         ∂Wmid       ∂Amid      ∂Zmid     ∂Wmid
        
        Etotal对Amid的偏导再细化
        
        ∂Etotal     ∂Etotal     ∂Aout     ∂Zout
        ———————  =  ———————  *  —————  *  —————
         ∂Amid       ∂Aout      ∂Zout     ∂Amid
        
        因此总的就等于
        
        ∂Etotal     ∂Etotal     ∂Aout     ∂Zout     ∂Amid     ∂Zmid
        ———————  =  ———————  *  —————  *  —————  *  —————  *  —————
         ∂Wmid       ∂Aout      ∂Zout     ∂Amid     ∂Zmid     ∂Wmid
        
        目前已知的是括号中的，是更新Wout时已经计算好的，现在只需要计算后面三个
        
        ∂Etotal       ∂Etotal     ∂Aout       ∂Zout     ∂Amid     ∂Zmid
        ———————  =  ( ———————  *  ————— )  *  —————  *  —————  *  —————
         ∂Wmid         ∂Aout      ∂Zout       ∂Amid     ∂Zmid     ∂Wmid
        
        ∂Etotal               ∂Zout     ∂Amid     ∂Zmid
        ———————  =  Et_zo  *  —————  *  —————  *  —————
         ∂Wmid                ∂Amid     ∂Zmid     ∂Wmid
        
        Zout对Amid的偏导是Wout，Amid对Zmid的偏导是Amid * (1 - Amid)，Zmid对Wmid的偏导是input（即输入的值）
        
        综上所述，结果为
        
        ∂Etotal
        ———————  =  Et_zo * Wout * (Amid * (1 - Amid)) * input
         ∂Wmid
       
        '''

        # w_mid2out是out行mid列，Et_zo是out行1列，结果Et_am是mid行1列
        Et_am = np.dot(self.w_mid2out.T, Et_zo)
        # print('Et_am.shape', Et_am.shape)

        # 计算w_in2mid更新所需要的参数
        am_zm = (Amid * (1 - Amid)).reshape(-1, 1)  # Amid对Zmid求偏导，Amid是mid行1列，结果am_zm也是mid行1列
        # print('am_zm.shape', am_zm.shape)

        # Et_am和am_zm都是mid行1列，结果也是mid行1列
        mid_error = Et_am * am_zm  # 为了保持原始w_in2mid的形状，所以必须是mid行，所以am_zm必须保留特征。因此w_mid2out与Et_zo计算必须算出一个mid行1列的矩阵
        # print('mid_error.shape', mid_error.shape)

        # 最终的形状要保持与w_in2mid相同的mid行in列
        Et_wm = np.dot(mid_error, train_list.T)  # train_list是in行1列，因此为了对应形状，需要转置train_list，再让竖着点乘横着的，就能乘出来mid行in列的矩阵了

        # 计算最终权重更新
        self.w_mid2out = self.w_mid2out - self.lrate * Et_wo
        # print(self.w_mid2out[0][:5])

        self.w_in2mid = self.w_in2mid - self.lrate * Et_wm
        # print(self.w_in2mid[0][:5])
        # print('================================')

    def query(self, inputs):
        Zmid = np.dot(self.w_in2mid, inputs)  # input → mid
        Amid = self.activation_function(Zmid)  # mid → sigmoid
        Zout = np.dot(self.w_mid2out, Amid)  # mid → output
        Aout = self.activation_function(Zout)  # output → sigmoid  Aout是out行1列
        return Aout

'''
调用mnist_train.csv读取训练数据，规范化训练数据
'''
with open('dataset/mnist_train.csv', 'r') as f:
    data_list1 = f.read().splitlines()
data_list1 = np.reshape(data_list1, (len(data_list1), 1))
# print(data_list.shape)

data_train = []  # 定义一个空组
for i in data_list1:
    # print(i)
    tmpl = []  # 定义一个临时空组
    tmp = i[0].split(',')  # 将data_list中取出的每一个元素以逗号为标识，分割这个元素，存入临时组tmp
    tmpl.append(int(tmp[0]))  # 将tmp第一个值，即答案，先int化，再存到tmpl列表中
    tmpl.extend(np.array(tmp[1:]).astype(np.int32) / 255 * 0.99 + 0.01)  # 将tmp后面的元素，转成int类型，进行特殊的归一化计算，存入临时组tmpl
    data_train.append(tmpl)  # 因为存的是临时组，因此将临时组存入data_train后，就是一组一组的形式了
train_list = np.array(data_train)
# print(train_list)
# print(train_list.shape)


'''
读取测试数据，调用mnist_test.csv
'''
with open('dataset/mnist_test.csv', 'r') as f:
    data_list2 = f.read().splitlines()
data_list2 = np.reshape(data_list2, (len(data_list2), 1))
# print(data_list.shape)

data_test = []  # 定义一个空组
for i in data_list2:
    # print(i)
    tmpl = []  # 定义一个临时空组
    tmp = i[0].split(',')  # 将data_list中取出的每一个元素以逗号为标识，分割这个元素，存入临时组tmp
    tmpl.append(int(tmp[0]))  # 将tmp第一个值，即答案，先int化，再存到tmpl列表中
    tmpl.extend(np.array(tmp[1:]).astype(np.int32) / 255 * 0.99 + 0.01)  # 将tmp后面的元素，转成int类型，进行特殊的归一化计算，存入临时组tmpl
    data_test.append(tmpl)  # 因为存的是临时组，因此将临时组存入data_test后，就是一组一组的形式了
test_list = np.array(data_test)
# print(test_list)
# print(test_list.shape)


'''
设置模型初始化时需要传入的参数值，并实例化，初始化模型
'''
input_nodes = 784
mid_nodes = 90
output_nodes = 10
learning_rate = 0.6

NN = NeuralNetwork(input_nodes, mid_nodes, output_nodes, learning_rate)

'''
用训练集训练权重
'''
epochs = 10
np.random.shuffle(train_list)
# print(train_list.shape)
# print(train_list)

# print('data_list[0][1:].shape', train_list[0][1:].shape)
# label[int(data_list[3][0])] = 0.99
# print(label)

for _ in range(epochs):
    for i in train_list:
        label_train = np.zeros((10, 1)) + 0.01
        train = i[1:].reshape(-1, 1)
        # print(train.shape)
        label_train[int(i[0])] = 0.99
        # print('label_list.shape', label_list.shape)
        NN.train(train, label_train)

'''
用训练出来的权重，在测试集测试一下
'''
result = []
np.random.shuffle(test_list)
for i in test_list:
    test = i[1:].reshape(-1, 1)
    output = NN.query(test)
    label = np.argmax(output)
    print('实际结果',i[0])
    print('网络结果',label)
    print(output)
    print('==================')
    if label == i[0]:
        result.append(1)
    else:
        result.append(0)
print('-----------------------------------------------')
result = np.array(result)
print('正确个数：', np.sum(result))
























