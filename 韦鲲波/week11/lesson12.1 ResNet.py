from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, AveragePooling2D, Flatten, Activation, concatenate
from keras.models import Model
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np


def block(inp, filters, block_type='Identity', conv1=None):
    data = Conv2D(
        filters=filters[0],
        kernel_size=1,
        strides=1,
    )(inp)
    data = BatchNormalization()(data)
    data = Activation('relu')(data)


    data = Conv2D(
        filters=filters[1],
        kernel_size=3,
        strides=1 if conv1 or block_type == 'Identity' else 2,
        padding='same',
    )(data)
    data = BatchNormalization()(data)
    data = Activation('relu')(data)


    data = Conv2D(
        filters=filters[2],
        kernel_size=1,
        strides=1,
    )(data)
    data = BatchNormalization()(data)


    if block_type == 'Identity':
        output = layers.add([inp, data])

    elif block_type == 'Conv':
        sc = Conv2D(
            filters=filters[2],
            kernel_size=1,
            strides=1 if conv1 else 2,
            padding='same',
        )(inp)
        sc = BatchNormalization()(sc)
        output = layers.add([data, sc])

    output = Activation('relu')(output)
    return output

def resnet50(inp_shape=(224, 224, 3)):
    img_input = Input(shape=inp_shape)

    # preprocess
    x = ZeroPadding2D(padding=(3, 3))(img_input)
    x = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    print(x.shape)
    x = MaxPooling2D(
        (3, 3),
        strides=(2, 2),
        padding='SAME'  # 只有用same模式的padding才能从112 pooling到56上
    )(x)
    print(x.shape)
    # stage 1
    x = block(x, [64, 64, 256], block_type='Conv', conv1=True)
    x = block(x, [64, 64, 256], block_type='Identity')
    x = block(x, [64, 64, 256], block_type='Identity')
    print(x.shape)
    # stage 2
    x = block(x, [128, 128, 512], block_type='Conv')
    x = block(x, [128, 128, 512], block_type='Identity')
    x = block(x, [128, 128, 512], block_type='Identity')
    x = block(x, [128, 128, 512], block_type='Identity')
    print(x.shape)
    # stage 3
    x = block(x, [256, 256, 1024], block_type='Conv')
    x = block(x, [256, 256, 1024], block_type='Identity')
    x = block(x, [256, 256, 1024], block_type='Identity')
    x = block(x, [256, 256, 1024], block_type='Identity')
    x = block(x, [256, 256, 1024], block_type='Identity')
    x = block(x, [256, 256, 1024], block_type='Identity')
    print(x.shape)
    # stage 4
    x = block(x, [512, 512, 2048], block_type='Conv')
    x = block(x, [512, 512, 2048], block_type='Identity')
    x = block(x, [512, 512, 2048], block_type='Identity')
    print(x.shape)
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(1000, activation='softmax')(x)

    model = Model(inputs=img_input, outputs=x)

    return model


if __name__ == '__main__':
    # 初始化模型
    model = resnet50()
    # 读取权重
    model.load_weights(r'resnet50_tf\resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    # 打印各层信息
    model.summary()

    # 读取图片
    # img_path = r'resnet50_tf\elephant.jpg'
    img_path = r'resnet50_tf\bike.jpg'

    # 数据做预处理
    img = load_img(img_path, target_size=(224, 224))
    data = img_to_array(img)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)

    # 开始推理
    prediction = model.predict(data)
    # 打印结果
    for i in decode_predictions(prediction)[0]:
        print(i)
