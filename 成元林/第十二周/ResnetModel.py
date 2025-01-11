import numpy as np
from keras import layers

from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Activation, BatchNormalization, \
    Flatten, Input
from keras.models import Model

from keras.preprocessing import image

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def cov_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    """
    残差网络结构卷积块
    @param input_tensor: 输入张量
    @param kernelsize: 卷积核大小
    @param filters: 通道列表
    @param strides: 步长
    @return:
    """
    re = Conv2D(filters[0], (1, 1), strides=strides, padding="valid")(input_tensor)
    re = BatchNormalization()(re)
    re = Activation("relu")(re)

    re = Conv2D(filters[1], kernel_size=kernel_size, strides=(1, 1), padding="same")(re)
    re = BatchNormalization()(re)
    re = Activation("relu")(re)

    re = Conv2D(filters[2], (1, 1), strides=(1, 1), padding="valid")(re)
    re = BatchNormalization()(re)

    other = Conv2D(filters[2], (1, 1), strides=strides, padding="valid")(input_tensor)
    other = BatchNormalization()(other)

    result = layers.add([re, other])
    re = Activation("relu")(result)
    return re


def identity_block(input_tensor, kernel_size, filters, strides=(1, 1)):
    re = Conv2D(filters[0], (1, 1), strides=(1, 1), padding="valid")(input_tensor)
    re = BatchNormalization()(re)
    re = Activation("relu")(re)

    re = Conv2D(filters[1], kernel_size=kernel_size, strides=(1, 1), padding="same")(re)
    re = BatchNormalization()(re)
    re = Activation("relu")(re)

    re = Conv2D(filters[2], (1, 1), strides=(1, 1), padding="valid")(re)
    re = BatchNormalization()(re)

    result = layers.add([re, input_tensor])
    re = Activation("relu")(result)
    return re


def ResNetModel(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)
    zero_in = ZeroPadding2D((3, 3))(img_input)
    conv_re = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="valid")(zero_in)
    ba_re = BatchNormalization()(conv_re)
    re = Activation("relu")(ba_re)
    pool_re = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(re)  # 输出 (c,h,w)=(64,56,56)

    # stage1
    stage1_cov = cov_block(pool_re, (3, 3), [64, 64, 256], strides=1)
    stage1_identity1 = identity_block(stage1_cov, (3, 3), [64, 64, 256], strides=1)
    stage1_identity2 = identity_block(stage1_identity1, (3, 3), [64, 64, 256], strides=1)

    # stage2
    stage2_cov = cov_block(stage1_identity2, (3, 3), [128, 128, 512], strides=2)
    stage2_identity1 = identity_block(stage2_cov, (3, 3), [128, 128, 512], strides=1)
    stage2_identity2 = identity_block(stage2_identity1, (3, 3), [128, 128, 512], strides=1)
    stage2_identity3 = identity_block(stage2_identity2, (3, 3), [128, 128, 512], strides=1)

    # stage3
    stage3_cov = cov_block(stage2_identity3, (3, 3), [256, 256, 1024], strides=2)
    stage3_identity1 = identity_block(stage3_cov, (3, 3), [256, 256, 1024], strides=1)
    stage3_identity2 = identity_block(stage3_identity1, (3, 3), [256, 256, 1024], strides=1)
    stage3_identity3 = identity_block(stage3_identity2, (3, 3), [256, 256, 1024], strides=1)
    stage3_identity4 = identity_block(stage3_identity3, (3, 3), [256, 256, 1024], strides=1)
    stage3_identity5 = identity_block(stage3_identity4, (3, 3), [256, 256, 1024], strides=1)

    # stage4
    stage4_cov = cov_block(stage3_identity5, (3, 3), [512, 512, 2048], strides=2)
    stage4_identity1 = identity_block(stage4_cov, (3, 3), [512, 512, 2048], strides=1)
    stage4_identity2 = identity_block(stage4_identity1, (3, 3), [512, 512, 2048], strides=1)
    # 平均池化
    final_pool = AveragePooling2D(pool_size=(7, 7))(stage4_identity2)
    # 展平
    re = Flatten()(final_pool)
    # 全连接
    re = Dense(classes, activation='softmax', name='fc1000')(re)
    model = Model(img_input, re, name="resnet50")

    return model


if __name__ == '__main__':
    model = ResNetModel([224, 224, 3], 1000)
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    # 会以表格形式打印出整个模型的层级结构。每一层都有一个编号，从0开始，依次递增。例如，(None, 32, 32, 3)表示输入形状为(None, 32, 32, 3)，
    # Sorted表示批量大小未指定，32x32x3表示图像的宽度、高度和通道数‌1。
    summary = model.summary()
    image_path = "bike.jpg"
    img = image.load_img(image_path, target_size=(224, 224))
    # 函数‌主要用于将PIL Image实例转换为Numpy数组，并且确保返回的是一个3D Numpy数组，无论输入图像是2D还是3D形状‌1。
    # 这个函数在处理图像数据时非常有用，尤其是在进行图像分类任务时，能够显著提升网络的性能‌
    xarray = image.img_to_array(img)
    x = np.expand_dims(xarray, axis=0)
    # 功能和作用
    # ‌调整图像数据格式‌：preprocess_input()函数会将输入的图像数据调整为适合Keras模型输入的格式。这包括调整图像的通道顺序，例如将RGB转换为BGR等‌1。
    # ‌标准化图像数据‌：该函数会对图像数据进行标准化处理，通常将像素值缩放到[-1, 1]或[0, 1]的范围内。标准化有助于减少数据的偏差，提高模型的收敛速度和准确性‌
    x = preprocess_input(x)

    predict = model.predict(x)
    print('Predicted1000:', predict)
    print('Predicted decode:', decode_predictions(predict))
