import numpy
import scipy.special

class NeuralNetWork(object):
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        #初始化权重
        """
        numpy.random.normal(mu,sigma,n):均值，标准差，输出形状大小
        """
        # 输入层至隐藏层
        self.wih = (numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))) #200*784
        # 隐藏层至输出层
        self.who = (numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))) #10*200

        #初始化激活函数sigma
        self.activation_fuction = lambda x:scipy.special.expit(x)

    [2]#训练 训练数据和正确答案
    def train(self,input_lists,out_lists):
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置,将输入转换为列向量
         ndmin=2来限制输入序列的最低维数是2维
         举个例子，如果input_list=[1,2,3],那么np.array(input_list,ndmin=2).T 将会返回一个形状为(3,1)的二维数组，
          即array([[1],
                  [2],
                 [3]])

        '''
        inputs = numpy.array(input_lists,ndmin=2).T
        targets = numpy.array(out_lists,ndmin=2).T
        hiden_inputs = numpy.dot(self.wih,inputs)#(200*784) * (784*1)
        hiden_outputs = self.activation_fuction(hiden_inputs)
        final_inputs = numpy.dot(self.who,hiden_outputs)#(10*200) * (200*1)
        final_outputs = self.activation_fuction(final_inputs)

        #计算误差
        output_errors = targets-final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors) #(10,200).T * (10*1)
        self.who += self.lr * numpy.dot(output_errors*final_outputs*(1-final_outputs),hidden_errors.T)# (10*1) * (200*1).T=10*200
        self.wih += self.lr * numpy.dot(hidden_errors*hiden_outputs*(1-hiden_outputs),inputs.T)# (200*1) * (784*1).T=200*784

        pass


    [3]#推理
    def query(self,inputs):
        hiden_inputs = numpy.dot(self.wih,inputs)
        hiden_outputs = self.activation_fuction(hiden_inputs)
        final_inputs = numpy.dot(self.who,hiden_outputs)
        final_outputs = self.activation_fuction(final_inputs)
        print(final_outputs)
        return final_outputs

#初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
inputnodes = 784
hiddennodes = 200
outputnodes = 10
learningrate = 0.1
n =NeuralNetWork(inputnodes, hiddennodes, outputnodes,learningrate)

#读入训练数据
"""open函数里的路径根据数据存储的路径来设定
   open(,'r')--只读模式
"""
trainning_data_file = open('dataset/mnist_train.csv','r')
trainning_data_list = trainning_data_file.readlines()
trainning_data_file.close()

#加入epochs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    #第一行，第二行，...
    for record in trainning_data_list:
        #因为csv文件是以','分隔的
        all_values = record.split(',')
        #数据归一化
        """
        从绘制的结果看，数据代表的确实是一个黑白图片的手写数字。
        数据读取完毕后，我们再对数据格式做些调整，以便输入到神经网络中进行分析。
        我们需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。
        由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
        有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题(因为权重都是0还更新啥呀，以及很多的乘法，0乘任何数都是0，所以不希望有那么多0)。
        所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
        
        numpy.asfarray()将输入数据转换为浮点型数组。[1:]，因为第一位是正确答案，所以从第二位开始
        """
        inputs = numpy.asfarray(all_values[1:])/255*0.99 + 0.01
        #将输入数据与正确答案对应好关系
        targets = numpy.zeros(outputnodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

#数据测试
test_data_file = open('dataset/mnist_test.csv','r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scores=[]

for read in test_data_list:
    all_values = read.split(',')
    #预处理数字图片
    inputs = numpy.asfarray(all_values[1:])/255.0*0.99 + 0.01
    correct_number = int(all_values[0]) #要加int()
    print('图片的正确答案是：',correct_number)
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    print('图片推理结果是：',label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
"""
numpy.asarray():可以将各种形式的输入数据转换为NumPy数组
"""
scores_array = numpy.asarray(scores)
print('该网络训练的正确率是：',scores_array.sum()/scores_array.size)















