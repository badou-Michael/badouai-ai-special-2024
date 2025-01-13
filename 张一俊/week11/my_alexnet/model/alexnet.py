# alexnet网络结构的构建

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# 为了加快收敛，将每个卷积层的filter减半，全连接层减为1024
def AlexNet(input_shape=(227, 227, 3), output_shape=2):
    model = Sequential()  # 顺序模型的网络结构

    # 第一层卷积层
    model.add(Conv2D(48, kernel_size=(11, 11), strides=(4, 4), padding='valid', input_shape=input_shape, activation='relu'))  # (55, 55, 48)
    model.add(BatchNormalization())  # 批量标准化，对每一层的输出进行标准化处理，使其均值为 0，方差为 1；用于加速训练过程，并提高神经网络的表现。
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))  # (27, 27, 48)

    # 第二层卷积层
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))  # (27, 27, 128)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))  # (13, 13, 128)

    # 第三、四、五层卷积层
    model.add(Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))  # (13, 13, 192)
    model.add(Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))  # (13, 13, 192)
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))  # (13, 13, 128)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))  # (6, 6, 128)

    # 展平并加入全连接层,一维向量（大小为 4608）会被传递给 1024 个神经元
    model.add(Flatten())  # 4608
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))  # 随机丢弃神经元，防止过拟合..

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 输出层
    model.add(Dense(output_shape, activation='softmax'))

    return model



# 卷积层的输出形状由以下几个因素决定：
#
# 输入形状：输入图像的尺寸 (H_in, W_in, C_in)，其中 H_in 是高度，W_in 是宽度，C_in 是通道数（深度）。
# 卷积核大小：卷积核的大小 (K_h, K_w)，分别是高度和宽度。
# 步长：卷积核在每一维上移动的步幅 (S_h, S_w)，分别是高度和宽度方向上的步长。
# 填充方式：通常有两种填充方式：
# valid：不填充，输出大小减小。
# same：填充，使得输出与输入的宽度和高度相同（在步长为 1 的情况下）。
# 卷积层的输出形状计算公式：
#
# 输出高度（H_out）和输出宽度（W_out）的计算公式为：
# Hout = (Hin - Kh + 2*padding.h)/S.h + 1
# W同理。把上述H换成W w就行。
#
# 其中：
# H_in 和 W_in 是输入的高度和宽度。
# K_h 和 K_w 是卷积核的高度和宽度。
# S_h 和 S_w 是步长。
# padding_h 和 padding_w 是填充的高度和宽度（如果使用 same 填充，计算后一般是自动决定的）。
# 注意：输出的高度和宽度必须是整数，所以在实际计算时，通常会向下取整，除非采用 same 填充。
#
# 以第二层卷积层为例（Conv2D）：
# 第二层卷积层的参数如下：
#
# 输入形状（input_shape）： (27, 27, 48)（从第一层卷积和池化输出）
# 卷积核大小（kernel_size）： (5, 5)
# 步长（strides）： (1, 1)
# 填充（padding）： 'same'（即填充使得输出与输入在宽度和高度上相同）
# 计算输出形状：
# 高度和宽度：
# 使用 same 填充时，输入的高度和宽度保持不变，输出的形状与输入相同。
# 输入高度和宽度是 27，卷积核大小是 5x5，步长是 1，采用 same 填充，因此输出高度和宽度仍然为 27。
# 通道数：
# 卷积层输出的通道数等于卷积核的数量（filters），在本例中为 128。

# 池化：
# 输出Ho = (Hin - Kh)/Sh + 1