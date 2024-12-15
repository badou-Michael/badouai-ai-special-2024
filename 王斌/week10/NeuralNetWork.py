import numpy as np
import scipy.special

class NeuralNetWork():
    def __init__(self,inputNodes,hiddenNodes,outputNodes,learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate

        self.wih = np.random.rand(self.hiddenNodes,self.inputNodes)-0.5
        self.who = np.random.rand(self.outputNodes,self.hiddenNodes)-0.5

        self.activation_function = lambda x:scipy.special.expit(x)

        pass

    def train(self,inputs_list,labels_list):
        inputs_list = np.array(inputs_list,ndmin=2).T
        labels_list = np.array(labels_list,ndmin=2).T

        hidden_input = np.dot(self.wih,inputs_list)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation_function(final_input)

        output_errors = labels_list-final_output
        hidden_errors = np.dot(self.who.T,output_errors*final_output*(1-final_output))
        self.who += self.learningRate*np.dot(output_errors*final_output*(1-final_output),np.transpose(hidden_output))
        self.wih += self.learningRate*np.dot(hidden_errors*hidden_output*(1-hidden_output),np.transpose(inputs_list))

    def query(self,inputNodes):
        hidden_input = np.dot(self.wih,inputNodes)
        hidden_output = self.activation_function(hidden_input)
        final_input = np.dot(self.who,hidden_output)
        final_output = self.activation_function(final_input)
        print(final_output)
        return final_output



input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learningRate = 0.3
n = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learningRate)

train_data_file = open("C:/Users/Administrator/Desktop/dataset/mnist_train.csv")
train_data_list = train_data_file.readlines()
train_data_file.close()
epochs = 10
for i in range(epochs):
    for record in  train_data_list:
        all_values = record.split(",")
        input = (np.asfarray(all_values[1:]))/255*0.99+0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(input,targets)

test_data_file = open("C:/Users/Administrator/Desktop/dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in  test_data_list:
    all_values = record.split(",")
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    input = np.asfarray(all_values[1:])/255*0.99 + 0.01
    outputs = n.query(input)
    label = np.argmax(outputs)
    print("网络输出的图片为：",label)
    if correct_number == label:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

scores_array = np.asfarray(scores)
print("perfermance=", scores_array.sum()/scores_array.size)

