from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

# 注意，为了加快收敛，我将每个卷积层的filter减半，全连接层减为1024
def AlexNet(input_shape=(224,224,3),output_shape=2):
    # AlexNet
    model = Sequential()
    '''
    Keras中，Sequential 是一个用于创建线性堆叠的神经网络模型的类。当你调用 model = Sequential() 时，你创建了一个空的模型，
    之后可以按顺序添加层（layers）。每添加一层，都会成为模型的一部分，并在模型中按顺序排列。
    '''
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    model.add(
        Conv2D(
            filters=48, 
            kernel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    # 所建模型后输出为48特征层
    model.add(
        MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128, 
            kernel_size=(5,5), 
            strides=(1,1), 
            padding='same',
            activation='relu'
        )
    )
    
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192, 
            kernel_size=(3,3),
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    ) 
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128, 
            kernel_size=(3,3), 
            strides=(1,1), 
            padding='same', 
            activation='relu'
        )
    )
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    # 所建模型后输出为128特征层
    model.add(
        MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )
    # 两个全连接层，最后输出为1000类,这里改为2类（猫和狗）
    # 缩减为1024
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))#1024表示这一全连接层有1024个神经元。每个神经元都会接收上一层所有神经元的输出，并计算加权和，然后通过激活函数（在这个例子中是ReLU）进行处理。
    model.add(Dropout(0.25))#0.25：丢弃神经元的比例，即25%的神经元在训练过程中会被随机丢弃。
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    '''
    这部分代码与前面的全连接层和Dropout层相同，再次添加一个具有1024个神经元的全连接层，使用ReLU激活函数，并添加一个丢弃25%神经元的Dropout层。
    目的：通过增加网络深度来提高模型的学习能力。
    
    output_shape：输出层的神经元数量，通常等于分类任务中的类别数。
    activation='softmax'：激活函数，使用Softmax函数。Softmax函数将输出转换为概率分布，适用于多分类问题。
    '''
    model.add(Dense(output_shape, activation='softmax')) #构建完所有层后，返回构建好的模型对象，以便后续的编译、训练和评估

    return model
'''
Flatten() 层：这个层的作用是将多维的输入一维化。通常在卷积神经网络（CNN）的最后使用，将卷积层的多维输出展平成一维数组，以便可以被全连接层（Dense layers）处理。
'''