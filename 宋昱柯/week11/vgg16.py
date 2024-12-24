import tensorflow as tf


class VGG16:
    def __init__(self, num_classes=1000, is_training=True, dropout_rate=0.5):
        """
        初始化VGG16网络
        参数:
            num_classes: 分类数量
            is_training: 是否为训练模式
            dropout_rate: dropout丢弃率
        """
        self.num_classes = num_classes
        self.is_training = is_training
        self.dropout_rate = dropout_rate
        self.slim = tf.contrib.slim

    def create_conv_layer(self, inputs, filters, repeat_times, scope_name):
        """
        创建卷积层块
        参数:
            inputs: 输入数据
            filters: 卷积核数量
            repeat_times: 重复次数
            scope_name: 作用域名称
        """
        with tf.variable_scope(scope_name):
            net = inputs
            # 重复执行卷积操作
            for i in range(repeat_times):
                net = self.slim.conv2d(net, filters, [3, 3], scope=f"conv_{i+1}")
            # 最大池化
            net = self.slim.max_pool2d(net, [2, 2], scope="pool")
            return net

    def create_fc_layer(
        self,
        inputs,
        output_size,
        kernel_size,
        use_dropout=True,
        is_last_layer=False,
        scope_name="fc",
    ):
        """
        创建全连接层（使用卷积实现）
        参数:
            inputs: 输入数据
            output_size: 输出维度
            kernel_size: 卷积核大小
            use_dropout: 是否使用dropout
            is_last_layer: 是否为最后一层
            scope_name: 作用域名称
        """
        with tf.variable_scope(scope_name):
            # 对于最后一层，不使用激活函数和归一化
            if is_last_layer:
                net = self.slim.conv2d(
                    inputs,
                    output_size,
                    kernel_size,
                    activation_fn=None,
                    normalizer_fn=None,
                )
            else:
                net = self.slim.conv2d(
                    inputs, output_size, kernel_size, padding="VALID"
                )

            # 添加dropout层
            if use_dropout and not is_last_layer:
                net = self.slim.dropout(
                    net, keep_prob=1 - self.dropout_rate, is_training=self.is_training
                )
            return net

    def build_model(self, inputs):
        """
        构建VGG16网络模型
        参数:
            inputs: 输入数据 (预期尺寸: 224x224x3)
        """
        with tf.variable_scope("vgg_16"):
            # 第一个卷积块：2个卷积层，64个卷积核
            # 输入: 224x224x3 -> 输出: 112x112x64
            net = self.create_conv_layer(inputs, 64, 2, "conv_block1")

            # 第二个卷积块：2个卷积层，128个卷积核
            # 输入: 112x112x64 -> 输出: 56x56x128
            net = self.create_conv_layer(net, 128, 2, "conv_block2")

            # 第三个卷积块：3个卷积层，256个卷积核
            # 输入: 56x56x128 -> 输出: 28x28x256
            net = self.create_conv_layer(net, 256, 3, "conv_block3")

            # 第四个卷积块：3个卷积层，512个卷积核
            # 输入: 28x28x256 -> 输出: 14x14x512
            net = self.create_conv_layer(net, 512, 3, "conv_block4")

            # 第五个卷积块：3个卷积层，512个卷积核
            # 输入: 14x14x512 -> 输出: 7x7x512
            net = self.create_conv_layer(net, 512, 3, "conv_block5")

            # 第一个全连接层：4096个神经元
            # 输入: 7x7x512 -> 输出: 1x1x4096
            net = self.create_fc_layer(net, 4096, [7, 7], scope_name="fc6")

            # 第二个全连接层：4096个神经元
            # 输入: 1x1x4096 -> 输出: 1x1x4096
            net = self.create_fc_layer(net, 4096, [1, 1], scope_name="fc7")

            # 输出层：num_classes个神经元
            # 输入: 1x1x4096 -> 输出: 1x1xnum_classes
            net = self.create_fc_layer(
                net,
                self.num_classes,
                [1, 1],
                use_dropout=False,
                is_last_layer=True,
                scope_name="fc8",
            )

            # 压缩输出维度
            # 从 [batch_size,1,1,num_classes] 变为 [batch_size,num_classes]
            output = tf.squeeze(net, [1, 2], name="final_output")

            return output


def create_vgg16_model(inputs, num_classes=1000, is_training=True):
    """
    创建VGG16模型的便捷函数
    """
    model = VGG16(num_classes=num_classes, is_training=is_training)
    return model.build_model(inputs)
