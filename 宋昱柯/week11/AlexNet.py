from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D
from keras.layers import Flatten, Dropout, BatchNormalization


class AlexNetModel:
    def __init__(self, image_height=224, image_width=224, channels=3, num_classes=2):
        """
        初始化AlexNet模型
        参数:
            image_height: 图片高度
            image_width: 图片宽度
            channels: 图片通道数
            num_classes: 分类数量
        """
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.num_classes = num_classes
        self.model = Sequential()

    def add_conv_layer(
        self, filters, kernel_size, strides, padding="same", is_first_layer=False
    ):
        """
        添加卷积层
        参数:
            filters: 卷积核数量
            kernel_size: 卷积核大小
            strides: 步长
            padding: 填充方式
            is_first_layer: 是否是第一层
        """
        if is_first_layer:
            self.model.add(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    input_shape=(self.image_height, self.image_width, self.channels),
                    activation="relu",
                )
            )
        else:
            self.model.add(
                Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation="relu",
                )
            )

        # 添加批归一化层
        self.model.add(BatchNormalization())

    def add_pool_layer(self, pool_size=(3, 3), strides=(2, 2)):
        """
        添加池化层
        参数:
            pool_size: 池化窗口大小
            strides: 步长
        """
        self.model.add(
            MaxPooling2D(pool_size=pool_size, strides=strides, padding="valid")
        )

    def add_dense_layer(self, units, dropout_rate=0.25):
        """
        添加全连接层
        参数:
            units: 神经元数量
            dropout_rate: Dropout比率
        """
        self.model.add(Dense(units, activation="relu"))
        self.model.add(Dropout(dropout_rate))

    def build_model(self):
        """
        构建完整的AlexNet模型
        """
        # 第一个卷积块
        # 输入: 224x224x3 -> 输出: 55x55x48 -> 27x27x48
        self.add_conv_layer(
            filters=48,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding="valid",
            is_first_layer=True,
        )
        self.add_pool_layer()

        # 第二个卷积块
        # 输入: 27x27x48 -> 输出: 27x27x128 -> 13x13x128
        self.add_conv_layer(filters=128, kernel_size=(5, 5), strides=(1, 1))
        self.add_pool_layer()

        # 第三个卷积块
        # 输入: 13x13x128 -> 输出: 13x13x192
        self.add_conv_layer(filters=192, kernel_size=(3, 3), strides=(1, 1))

        # 第四个卷积块
        # 输入: 13x13x192 -> 输出: 13x13x192
        self.add_conv_layer(filters=192, kernel_size=(3, 3), strides=(1, 1))

        # 第五个卷积块
        # 输入: 13x13x192 -> 输出: 13x13x128 -> 6x6x128
        self.add_conv_layer(filters=128, kernel_size=(3, 3), strides=(1, 1))
        self.add_pool_layer()

        # 展平层
        self.model.add(Flatten())

        # 全连接层
        self.add_dense_layer(units=1024)
        self.add_dense_layer(units=1024)

        # 输出层
        self.model.add(Dense(self.num_classes, activation="softmax"))

        return self.model


def create_alexnet(image_size=(224, 224), channels=3, num_classes=2):
    """
    创建AlexNet模型的便捷函数
    参数:
        image_size: 输入图片尺寸
        channels: 图片通道数
        num_classes: 分类数量
    返回:
        构建好的AlexNet模型
    """
    height, width = image_size
    model_builder = AlexNetModel(
        image_height=height,
        image_width=width,
        channels=channels,
        num_classes=num_classes,
    )
    return model_builder.build_model()
