import numpy as np
import scipy.special

class NNW:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # Initial weight
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))  
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))  

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T  
        targets = np.array(targets_list, ndmin=2).T  
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T  
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# Initialize
inputnode = 28 * 28
hiddennode = 200
outputnode = 10
learningrate = 0.1
n = NNW(inputnode, hiddennode, outputnode, learningrate)

# Load training data
train_file = open("dataset/mnist_train.csv", 'r')
train_list = train_file.readlines()
train_file.close()

# Train the network
for i in range(5):  # Train for 5 epochs
    for record in train_list:
        values = record.split(',')
        inputs = (np.asarray(values[1:], dtype=float)) / 255.0 * 0.99 + 0.01
        targets = np.zeros(outputnode) + 0.01
        targets[int(values[0])] = 0.99
        n.train(inputs, targets)

# Load test data
test_data_file = open('dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Evaluate performance
score = []
for record in test_data_list:
    values = record.split(',')
    correct_number = int(values[0])
    inputs = (np.asarray(values[1:], dtype=float)) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(f'Correct number is: {correct_number}. Prediction number: {label}')
    if label == correct_number:
        score.append(1)
    else:
        score.append(0)

print(f'Score: {sum(score) / len(score) * 100}%')
