import numpy as np

def data_set():

    with open(r'alexnet\AlexNet-Keras-master\data\dataset.txt', 'r') as f:
        file = f.readlines()

    # 打乱file中的行
    np.random.shuffle(file)
    num_train = int(len(file) * 0.9)
    num_val = len(file) - num_train

    return num_train, num_val









