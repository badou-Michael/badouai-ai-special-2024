import numpy
import scipy.special

class NeuralNetWork:
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    #初始化网络，输入层、隐藏层和输出层节点数
    self.inodes = inputnodes
    self.hnodes = hiddenodes
    self.onodes = outputnodes

    #设置学习率
    self.lr = learningrate

    '''
    接下来初始化权重矩阵： 输入层和中间层节点间链路权重以及中间层和输出层间链路权重形成的矩阵
    '''
    self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
    self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

    '''
    激活函数的设置
    '''
    self.activation_function = lambda x:scipy.special.expit(x)

    pass

  def train(self, inputs_list, target_list):
    #根据输入的训练数据更新节点链路权重
    inputs = numpy.array(inputs_list, ndmin=2).T
    targets = numpy.array(targets_list, ndemin=2).T

    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)
    final_inputs = numpy.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)

    output_errors = targets - final_outputs
    hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs *(1 - final_outputs))

    self.who += self.lr * numpy.dot((output_errors * final_outputs*(1 - final_outputs)),
                                    numpy.transpose(hidden_outputs))
    self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                    numpy.transpose(inputs))
    pass
  
  def query(self, inputs):
    hidden_inputs = numpy.dot(self.wih, inputs)
    hidden_outputs = self.activation_function(hidden_inputs)
    final_inputs = numpy.dot(self.who, hidden_outputs)
    final_outputs = self.activation_function(final_inputs)
    print(final_outputs)
    return final_outputs

#初始化网络参数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
  for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

test_data_file = open('dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data.file.close()
scores = []
for record in test_data_list:
  all_values = record.split(',')
  correct_number = int(all_values[0])
  print('该图片对应的数字为:', correct_number)
  inputs = (numpy.asfarray(all_values[1:]))/255*0.99+0.01
  outputs = n.query(inputs)
  label = numpy.argmax(outputs)
  print('网络认为图片的数字是 ', label)
  if label == correct_number:
    scores.append(1)
  else:
    scores.append(0)

print(scores)
scores_array = numpy.asarray(scores)
print('perfermance = ', scores_array.sum() / scores_array.size())







    
    



    
