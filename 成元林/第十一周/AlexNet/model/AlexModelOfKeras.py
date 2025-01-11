from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# 通道都减半
def AlexNet_cat_dog_Model(input_shape=(224, 224, 3), output_shape=2):
    """
    AlexNet猫狗图像识别模型
    @param input_shape: 输入形状
    @param output_shape: 输出形状
    @return:
    """
    # ‌1.创建Sequential模型
    model = Sequential()
    # 1.输入图片进行卷积操作，卷积核为11x11,strides=4,输出通道为96/2=48
    model.add(
        Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding="valid", data_format=None, activation="relu",
               input_shape=input_shape))
    model.add(BatchNormalization())
    # 最大池化
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))

    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    #铺平
    model.add(Flatten())
    #全连接
    model.add(Dense(units=1024,activation="relu"))
    model.add(Dropout(rate=0.25))
    model.add(Dense(1024,activation="relu"))
    model.add(Dropout(rate=0.25))
    model.add(Dense(output_shape,activation="softmax"))
    return model



