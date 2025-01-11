# 实现alexnet
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

def AlexNet_SELF(input_shape=(224,224,3),output_shape=2):
    model = Sequential()
    # 增加第一层卷积：卷积核尺寸为11，步长为4，卷积核个数96
    # 输出尺寸：55*55*96
    model.add(
        Conv2D(
            filters=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    # 增加一个BatchNormalization，归一化，可以减少训练时间，增加训练精确度
    model.add(BatchNormalization())
    #增加最大池化
    # 输出尺寸：27*27*96
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 增加第二层卷积：卷积核尺寸为5，步长为1，卷积核个数256
    # 输出尺寸：27*27*256
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    model.add(BatchNormalization())
    # 增加最大池化
    # 输出尺寸：13*13*256
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    # 增加第三层卷积：卷积核尺寸为3，步长为1，卷积核个数384
    # 输出尺寸：13*13*384
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 增加第四层卷积：卷积核尺寸为3，步长为1，卷积核个数384
    # 输出尺寸：13*13*384
    model.add(
        Conv2D(
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 增加第五层卷积：卷积核尺寸为3，步长为1，卷积核个数256
    # 输出尺寸：13*13*256
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 增加最大池化
    # 输出尺寸：6*6*256
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )
    #增加全连接层：先对卷积结果进行拍扁
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.25))

    #全连接+softmax，输出两类
    model.add(Dense(output_shape, activation='softmax'))

    return model
