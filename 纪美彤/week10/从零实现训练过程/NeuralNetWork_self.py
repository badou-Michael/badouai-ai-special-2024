import numpy
from sympy.stats.sampling.sample_scipy import scipy


# 我们的代码要导出三个接口，分别完成以下功能：
# 1，初始化initialisation，设置输入层，中间层，和输出层的节点数。
# 2，训练train:根据训练数据不断的更新权重值
# 3，查询query，把新的数据输入给神经网络，网络计算后输出答案。（推理）
class NeuralNetWork_self:
    # 1，初始化initialisation，设置输入层，中间层，和输出层的节点数。
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.input_nodes =inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes

        self.learning_rate = learningrate

        # 初始化权重矩阵，根据网络层数确定矩阵个数
        # numpy.random.rand初始化从0到1之间的数字，-0.5使初始化区间变为-0.5~0.5
        self.weight_input_hidden = numpy.random.rand(self.hidden_nodes,self.input_nodes) - 0.5
        self.weight_hidden_output = numpy.random.rand(self.output_nodes,self.hidden_nodes) - 0.5

        # 设置网络的激活函数
        # scipy.special.expit即sigmoid函数
        self.activation_function = lambda x:scipy.special.expit(x)

        pass

    def query(self, inputs):
        # 首先获得隐藏层的加权输入和
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        # 获得隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 获得输出层的加权输入和
        final_inputs = numpy.dot(self.weight_hidden_output,hidden_outputs)
        # 获得输出层结果
        final_outputs = self.activation_function(final_inputs)
        print("final_outputs = ",final_outputs)
        return final_outputs

        pass

    def train(self,input_list,target_list):
        #实现正向传播
        # 将input_list,target_list转换为numpy支持的二维矩阵
        inputs = numpy.array(input_list,ndmin = 2).T
        targets = numpy.array(target_list,ndmin = 2).T
        # 首先获得隐藏层的加权输入和
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        # 获得隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 获得输出层的加权输入和
        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        # 获得输出层结果
        final_outputs = self.activation_function(final_inputs)

        # 计算结果与目标的误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weight_hidden_output.T, output_errors*final_outputs*(1 - final_outputs))

        # 反向传播
        # 根据误差进行权重更新: 新权值 = 当前权值 - 学习率 × 梯度
        self.weight_hidden_output += self.learning_rate * numpy.dot((output_errors*final_outputs*(1 - final_outputs)),numpy.transpose(hidden_outputs))
        self.weight_input_hidden += self.learning_rate * numpy.dot((hidden_errors*hidden_outputs*(1 - hidden_outputs)),numpy.transpose(inputs))

        pass


# input_nodes = 3
# hidden_nodes = 3
# output_nodes = 3
#
# learning_rate = 0.3
# n = NeuralNetWork_self(input_nodes,hidden_nodes,output_nodes,learning_rate)
# n.query([1,0.5,-1.5])

# open函数里的路径根据数据存储的路径来设定
data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print("len(data_list) = ",len(data_list))
print("data_list[0] = ",data_list[0])

# 把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
# 第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

# 最外层有10个输出节点
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print("targets = ",targets)  # targets第8个元素的值是0.99，这表示图片对应的数字是7(数组是从编号0开始的).

'''
根据上述做法，我们就能把输入图片给对应的正确数字建立联系，这种联系就可以用于输入到网络中，进行训练。
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点。
这里需要注意的是，中间层的节点我们选择了100个神经元，这个选择是经验值。
中间层的节点数没有专门的办法去规定，其数量会根据不同的问题而变化。
确定中间层神经元节点数最好的办法是实验，不停的选取各种数量，看看那种数量能使得网络的表现最好。
'''
# 初始化网络
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetWork_self(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()
# 把数据依靠','区分，并分别读入


'''
在原来网络训练的基础上再加上一层外循环
但是对于普通电脑而言执行的时间会很长。
epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题.
'''
epochs = 100

for e in range(epochs):
    for record in trainning_data_list:
        all_values = record.split(',')
        # 归一化: 由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
        # 有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。
        # 所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


'''
最后我们把所有测试图片都输入网络，看看它检测的效果如何
'''
scores = []
for record in data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理数字图片，归一化
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字,推理
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的 编号
    label = numpy.argmax(outputs)
    print("output reslut is : ", label)
    # print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)


