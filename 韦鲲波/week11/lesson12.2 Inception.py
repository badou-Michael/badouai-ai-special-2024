from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, AveragePooling2D, Flatten, Activation, concatenate
from keras.models import Model
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np


def conv2d(inp, filters, kernel_size, strides=1, padding='same'):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
    )(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

class InceptionV3:
    def __init__(self):
        self.inp = Input(shape=(299, 299, 3))

    def __conv(self, inp):
            # input must 299*299*3
            x = conv2d(
                inp=inp,
                filters=32,
                kernel_size=3,
                strides=2,
                padding='valid',
            )
            x = conv2d(
                inp=x,
                filters=32,
                kernel_size=3,
                padding='valid',
            )
            x = conv2d(
                inp=x,
                filters=32,
                kernel_size=3,
                padding='same',
            )
            x = MaxPooling2D(
                pool_size=3,
                strides=2,
                padding='valid',
            )(x)
            x = conv2d(
                inp=x,
                filters=80,
                kernel_size=1,
                padding='valid',
            )
            x = conv2d(
                inp=x,
                filters=192,
                kernel_size=3,
                padding='valid',
            )
            x = MaxPooling2D(
                pool_size=3,
                strides=2,
                padding='valid',
            )(x)
            print(x.shape)
            return x

    def __module1(self, inp, chn):
            p1 = conv2d(
                inp=inp,
                filters=64,
                kernel_size=1,
                padding='same',
            )

            p2 = conv2d(
                inp=inp,
                filters=48,
                kernel_size=1,
                padding='same',
            )
            p2 = conv2d(
                inp=p2,
                filters=64,
                kernel_size=5,
                padding='same',
            )

            p3 = conv2d(
                inp=inp,
                filters=64,
                kernel_size=1,
                padding='same',
            )
            p3 = conv2d(
                inp=p3,
                filters=96,
                kernel_size=3,
                padding='same',
            )
            p3 = conv2d(
                inp=p3,
                filters=96,
                kernel_size=3,
                padding='same',
            )

            p4 = AveragePooling2D(
                pool_size=3,
                strides=1,
                padding='same',
            )(inp)
            p4 = conv2d(
                inp=p4,
                filters=chn,
                kernel_size=1,
                padding='same',
            )
            print(p1.shape, p2.shape, p3.shape, p4.shape)
            output = concatenate([p1, p2, p3, p4])

            return output

    def __module2(self, inp, chn):
            p1 = conv2d(
                inp=inp,
                filters=192,
                kernel_size=1,
                padding='same',
            )

            p2 = conv2d(
                inp=inp,
                filters=chn,
                kernel_size=1,
                padding='same',
            )
            p2 = conv2d(
                inp=p2,
                filters=chn,
                kernel_size=(1, 7),
                padding='same',
            )
            p2 = conv2d(
                inp=p2,
                filters=192,
                kernel_size=(7, 1),
                padding='same',
            )

            p3 = conv2d(
                inp=inp,
                filters=chn,
                kernel_size=1,
                padding='same',
            )
            p3 = conv2d(
                inp=p3,
                filters=chn,
                kernel_size=(7, 1),
                padding='same',
            )
            p3 = conv2d(
                inp=p3,
                filters=chn,
                kernel_size=(1, 7),
                padding='same',
            )
            p3 = conv2d(
                inp=p3,
                filters=chn,
                kernel_size=(7, 1),
                padding='same',
            )
            p3 = conv2d(
                inp=p3,
                filters=192,
                kernel_size=(1, 7),
                padding='same',
            )

            p4 = AveragePooling2D(
                pool_size=3,
                strides=1,
                padding='same',
            )(inp)
            p4 = conv2d(
                inp=p4,
                filters=192,
                kernel_size=1,
                padding='same',
            )
            print(p1.shape, p2.shape, p3.shape, p4.shape)
            output = concatenate([p1, p2, p3, p4])

            return output

    def __module3(self, inp):
        p1 = conv2d(
            inp=inp,
            filters=320,
            kernel_size=1,
            padding='same',
        )

        p2 = conv2d(
            inp=inp,
            filters=384,
            kernel_size=1,
            padding='same',
        )
        p2_1 = conv2d(
            inp=p2,
            filters=384,
            kernel_size=(1, 3),
            padding='same',
        )
        p2_2 = conv2d(
            inp=p2,
            filters=384,
            kernel_size=(3, 1),
            padding='same',
        )
        p2 = concatenate([p2_1, p2_2])

        p3 = conv2d(
            inp=inp,
            filters=448,
            kernel_size=1,
            padding='same',
        )
        p3 = conv2d(
            inp=p3,
            filters=384,
            kernel_size=3,
            padding='same',
        )
        P3_1 = conv2d(
            inp=p3,
            filters=384,
            kernel_size=(1, 3),
            padding='same',
        )
        P3_2 = conv2d(
            inp=p3,
            filters=384,
            kernel_size=(3, 1),
            padding='same',
        )
        p3 = concatenate([P3_1, P3_2])

        p4 = AveragePooling2D(
            pool_size=3,
            strides=1,
            padding='same',
        )(inp)
        p4 = conv2d(
            inp=p4,
            filters=192,
            kernel_size=1,
            padding='same',
        )
        print(p1.shape, p2.shape, p3.shape, p4.shape)
        output = concatenate([p1, p2, p3, p4])

        return output

    def __mid1(self, inp):
        p1 = conv2d(
            inp=inp,
            filters=384,
            kernel_size=3,
            strides=2,
            padding='valid',
        )

        p2 = conv2d(
            inp=inp,
            filters=64,
            kernel_size=1,
            padding='same',
        )
        p2 = conv2d(
            inp=p2,
            filters=96,
            kernel_size=3,
            padding='same',
        )
        p2 = conv2d(
            inp=p2,
            filters=96,
            kernel_size=3,
            strides=2,
            padding='valid',
        )

        p3 = AveragePooling2D(
            pool_size=3,
            strides=2,
            padding='valid',
        )(inp)
        print(p1.shape, p2.shape, p3.shape)
        output = concatenate([p1, p2, p3])

        return output

    def __mid2(self, inp):
        p1 = conv2d(
            inp=inp,
            filters=192,
            kernel_size=1,
            padding='same',
        )
        p1 = conv2d(
            inp=p1,
            filters=320,
            kernel_size=3,
            strides=2,
            padding='valid',
        )

        p2 = conv2d(
            inp=inp,
            filters=192,
            kernel_size=1,
            padding='same',
        )
        p2 = conv2d(
            inp=p2,
            filters=192,
            kernel_size=(1, 7),
            padding='same',
        )
        p2 = conv2d(
            inp=p2,
            filters=192,
            kernel_size=(7, 1),
            padding='same',
        )
        p2 = conv2d(
            inp=p2,
            filters=192,
            kernel_size=3,
            strides=2,
            padding='valid',
        )

        p3 = AveragePooling2D(
            pool_size=3,
            strides=2,
            padding='valid',
        )(inp)
        print(p1.shape, p2.shape, p3.shape)
        output = concatenate([p1, p2, p3])

        return output

    def perf(self):
        x = self.__conv(self.inp)

        x = self.__module1(x, chn=32)
        x = self.__module1(x, chn=64)
        x = self.__module1(x, chn=64)

        x = self.__mid1(x)

        x = self.__module2(x, chn=128)
        x = self.__module2(x, chn=160)
        x = self.__module2(x, chn=160)
        x = self.__module2(x, chn=192)

        x = self.__mid2(x)

        x = self.__module3(x)
        x = self.__module3(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(
            units=1000,
            activation='softmax',
        )(x)

        model = Model(inputs=self.inp, outputs=x)

        return model


def preprocessing(inp):
    x = load_img(inp)
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


if __name__ == '__main__':
    model = InceptionV3().perf()
    model.summary()
    model.load_weights(r'inceptionV3_tf\inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img = preprocessing(r'inceptionV3_tf/elephant.jpg')

    result = model.predict(img)
    print('结果：', decode_predictions(result))








