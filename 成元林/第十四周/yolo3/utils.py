import os.path

from PIL import Image
import config
import numpy as np
import tensorflow as tf

def resize_image(image, size):
    """
    对预测输入图像同比例缩放，不足地方填充
    @param image: 输入图像
    @param size: 图像大小
    @return: 缩放后的图像
    """
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))
    #BICUBIC,三次样条插值，计算量大，但效果更好‌，
    # BILINEAR：双线性插值
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)
    #不足的地方，填充
    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_image

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
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'conv2d' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'batch_normalization' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops

if __name__ == '__main__':
    # image = Image.open("./img/img2.jpg")
    # box = resize_image(image, (416, 416))
    # image.show()
    # anchors_path = os.path.expanduser(config.anchors_path)
    # anchors = []
    # with open(anchors_path) as f:
    #     anchor = f.readlines()
    # print(anchor)
    # anchors = [an.strip() for an in anchor]
    # print(anchors)
    anchors_path = os.path.expanduser(config.anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        # anchors = np.array(anchors).reshape(-1, 2)
    print(anchors)