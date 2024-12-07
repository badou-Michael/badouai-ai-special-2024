import numpy as np
import scipy.special
import cv2
class NeuralNetwork:
    def __init__(self, input_node, hidden_node, output_node, learning_rate):
        # Set up input layer, hidden layer, and output layer
        self.inodes = input_node
        self.hnodes = hidden_node
        self.onodes = output_node

        # Set up the learning rate
        self.lr = learning_rate

        # Initializing the weight matrix using Xavier initialization and normal distribution
        # since the activation function below is symmetric (sigmoid)
        # wih means the weight between input layer and hidden layer, iho means the weight between
        # hidden layer and output layer
        self.wih = np.random.normal(0.0, pow(self.inodes + self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes + self.onodes, -0.5), (self.onodes, self.hnodes))
        
        # Define activation function using sigmoid
        self.af = lambda x: scipy.special.expit(x)

        pass

    def train(self, input_list, target_list):
        # Update the weight matrix based on input data

        # Convert input_list and target_list to np arrays, with dimention 2, and transpose
        # so that it can multiply with the weight matrix
        input = np.array(input_list, ndmin = 2).T
        target = np.array(target_list, ndmin = 2).T

        # Calculate the input matrix for hidden layer, which is wih * input
        hidden_input = np.dot(self.wih, input)

        # Calculate the output for hidden layer by passing the input above to activation function
        hidden_output = self.af(hidden_input)

        # Calculate the input for output layer, which is who * hidden_output
        input_output = np.dot(self.who, hidden_output)

        # Calculate the output for output layer by passing input to activation function
        output_output = self.af(input_output)

        # Calculate the output errors and hidden errors
        output_error = target - output_output
        hidden_error = np.dot(self.who.T, output_error * output_output * (1 - output_output))

        # Update the weight matrixs
        self.who += self.lr * np.dot(output_error * output_output * (1 - output_output),
                                      hidden_output.T)
        self.wih += self.lr * np.dot(hidden_error * hidden_output * (1 - hidden_output),
                                     input.T)
        
        pass

    def query(self, input):
        # Get the answer based on input
        # Calculate the input for hidden layer
        hidden_input = np.dot(self.wih, input)

        # Calculate the output for hidden layer
        hidden_output = self.af(hidden_input)

        # Calculate the input for output layer
        output_input = np.dot(self.who, hidden_output)

        # Calculate the output for output layer
        output_output = self.af(output_input)
        return output_output

# Initializing
input_node = 28 * 28
hidden_node = 200
output_node = 10
learning_rate = 0.05
network = NeuralNetwork(input_node, hidden_node, output_node, learning_rate)

# Read training data
train_data = open("dataset/mnist_train.csv", 'r')
train_data_list = train_data.readlines()
train_data.close()


epochs = 30
for epoch in range(epochs):
    print("training epoch ", epoch)
    for record in train_data_list:
        # Convert input data to array wit value between 0 to 1 exclusive
        val = record.split(',')
        input = np.asfarray(val[1:])/255 * 0.99 + 0.01
        # Covert the target to array, the correct position is 0.99
        target = np.zeros(output_node) + 0.01
        target[int(val[0])] = 0.99
        network.train(input, target)

# Read in test data
test_data = open("dataset/mnist_test.csv")
test_data_list = test_data.readlines()
test_data.close()
scores = []

for record in test_data_list:
    val = record.split(',')
    correct_number = int(val[0])
    print("the correct number is ", correct_number)
    input = np.asfarray(val[1:]) / 255 * 0.99 + 0.01
    output = network.query(input)
    lable = np.argmax(output)
    print("the predicted number is ", lable)
    if lable == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)

scores_arr = np.asarray(scores)
print("perfermance = ", scores_arr.sum() / scores_arr.size)

# Testing
img = cv2.imread("dataset/my_own_4.png", cv2.IMREAD_GRAYSCALE)
input = img.flatten()
input = 1 - (np.asfarray(input) / 255*0.99 + 0.01)
output = network.query(input)
print(output)
print(np.argmax(output))
