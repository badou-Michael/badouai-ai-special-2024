import numpy as np
import tensorflow as tf
from PIL import Image

def load_weights(var_list, weights_file):
    """
    Introduction
    ------------
        加载预训练好的darknet53权重文件
    Parameters
    ----------
        var_list: 赋值变量名
        weights_file: 权重文件
    Returns
    -------
        assign_ops: 赋值更新操作
    """
    with open(weights_file, "rb") as fp:
        # 跳过文件的前五个整数值（通常是包含有关网络结构的一些信息）
        _ = np.fromfile(fp, dtype = np.int32, count = 5)
        weights = np.fromfile(fp, dtype=np.float32)
    
    ptr = 0
    i = 0
    assign_ops = []

    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i+1]
        # 如果当前层是卷积层 (conv2d)，则继续处理与卷积层相关的后续层
        if 'conv2d' in var1.name.split('/')[-2]: # 倒数第二个元素
            # 如果后面紧跟着批量归一化层（batch_normalization），则加载批量归一化层的参数
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i+1:i+5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape = True))
                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape = True))
                # we load 1 variable
                i += 1

            # 获取卷积层权重的形状, 例如：[kernel_height, kernel_width, in_channels, out_channels]）
            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape = True))
            i += 1

    return assign_ops

# 对预测输入图像进行缩放
def letterbox_image(image, size):
    """
    Introduction
    ------------
        对预测输入图像进行缩放,按照长宽比进行缩放,不足的地方进行填充
    Parameters
    ----------
        image: 输入图像
        size: 图像大小
    Returns
    -------
        boxed_image: 缩放后的图像
    """
    image_w, image_h = image.size
    w, h = size # 目标大小
    scale = min(w*1.0/image_w, h*1.0/image_h)
    new_w = int(image_w * scale)
    new_h = int(image_h * scale)
    # 三次样条插值 
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)
    # 新生成图片, 并默认灰色
    boxed_image = Image.new('RGB', size, (128,128,128))
    # 将缩放后的图像粘贴到新创建的图像中心位置
    # 指定 resized_image 在 boxed_image的偏移量
    boxed_image.paste(resized_image, ((w-new_w)//2, (h-new_h)//2))
    return boxed_image