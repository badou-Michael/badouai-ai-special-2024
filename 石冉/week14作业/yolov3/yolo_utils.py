from PIL import Image
import json
import numpy as np
import tensorflow as tf
from collections import defaultdict


#定义一个对图片进行缩放操作的函数，image是传入的图片，size是目标尺寸
def letterbox_image(image,size):
    image_w,image_h=image.size
    w,h=size
    new_w=int(image_w*min(w*1.0/image_w,h*1.0/image_h))
    new_h=int(image_h*min(w*1.0/image_w,h*1.0/image_h))
    resized_image=image.resize((new_w,new_h),Image.BICUBIC) #Image.BICUBIC:三次多项式插值
    boxed_image=Image.new('RGB',size,(128,128,128))#创建一个新的图像，尺寸为 size，填充颜色为灰色（128, 128, 128）
    boxed_image.paste(resized_image,((w-new_w)//2,(h-new_h)//2)) #将缩放后的图像粘贴到新图像的中心位置

    return boxed_image

#定义一个函数，用于加载预训练的Darknet53模型权重到指定的变量列表中
def load_weights(var_list,weights_file):
    '''introduction:加载预训练好的darknet53权重文件
    parameter：
    var_list:赋值变量名
    weights_file：权重文件
    return:
    assign_ops：赋值更新操作'''

    #以二进制读模式打开权重文件，并读取前5个整数（通常是Darknet权重文件的头信息）
    with open(weights_file,'rb') as fp:
        _=np.fromfile(fp,dtype=np.int32,count=5)
        #继续从文件中读取剩余的权重数据，数据类型为float32
        weights=np.fromfile(fp,dtype=np.float32)

    #初始化指针ptr用于跟踪权重数组中的当前位置，i用于遍历变量列表，assign_ops用于存储赋值操作
    ptr=0
    i=0
    assign_ops=[]
    #使用while循环遍历变量列表，每次处理两个连续的变量，var1和var2
    while i<len(var_list)-1:
        var1=var_list[i]
        var2=var_list[i+1]
        #检查当前处理的是否是卷积层（通过变量名中是否包含'conv2d'来判断）
        if 'conv2d' in var1.name.split('/')[-2]:
            #如果下一个层是批量归一化层（通过变量名中是否包含'batch_normalization'来判断）
            if 'batch_normalization' in var2.name.split('/')[-2]:
                #加载批量归一化参数，包括gamma、beta、mean和var。
                gamma,beta,mean,var=var_list[i+1,i+5]
                batch_norm_vars=[gamma,beta,mean,var]
                #对于每个批量归一化参数，计算其形状，从权重数组中读取相应的参数，并更新指针ptr，然后将参数赋值给对应的变量。
                for var in batch_norm_vars:
                    shape=var.shape.as_list()
                    num_params=np.prod(shape)
                    var_weights=weights[ptr:ptr+num_params].reshape(shape)
                    ptr+=num_params
                    assign_ops.append(tf.assign(var,var_weights,validate_shape=True))

                #由于已经加载了4个批量归一化参数，所以i增加4。
                i+=4
                #如果下一个层也是卷积层。
            elif 'conv2d' in var2.name.split('/')[-2]:
                #加载卷积层的偏置项，计算形状和参数数量，从权重数组中读取偏置参数，并更新指针ptr，然后将参数赋值给对应的变量。
                bias=var2
                bias_shape=bias.shape.as_list()
                bias_params=np.prod(bias_shape)
                bias_weights=weights[ptr:ptr+bias_params].reshape(shape)
                ptr+=bias_params
                assign_ops.append(tf.assign(bias,bias_weights,validate=True))
                i+=1 #由于已经加载了一个偏置项，所以i增加1

            #加载卷积层的权重，计算形状和参数数量
            shape=var1.shape.as_list()
            num_params=np.prod(shape)
            #从权重数组中读取卷积层的权重，调整形状，并转置为列主序（Darknet使用的是列主序），
            # 更新指针ptr，然后将参数赋值给对应的变量，i增加1。
            var_weights=weights[ptr:ptr+num_params].reshape((shape[3],shape[2],shape[0],shape[1]))
            var_weights=np.transpose(var_weights,(2,3,1,0))
            ptr+=num_params
            assign_ops.append(tf.assign(var1,var_weights,validate_shape=True))
            i+=1
    return assign_ops

#定义函数，在图像上绘制边界框（bounding boxes），并将结果通过TensorBoard进行可视化
def draw_box(image,bbox):
    '''introduction:通过tensorboard将训练结果可视化
    parameters：
    image:训练数据图片
    bbox:训练数据图片中标记box坐标'''
    #使用 tf.split 函数将 bbox 张量按照最后一个维度（axis=2）分割成5个部分，
    # 分别对应边界框的左上角x坐标（xmin）、左上角y坐标（ymin）、右下角x坐标（xmax）、右下角y坐标（ymax）和标签（label）。
    xmin,ymin,xmax,ymax,label=tf.split(value=bbox,num_or_size_splits=5,axis=2)
    #获取图像的高度和宽度，并将它们转换为float32类型，以便后续进行浮点数除法操作
    height=tf.cast(tf.shape(image)[1],tf.float32)
    weight=tf.cast(tf.shape(image)[2],tf.float32)
    #将边界框的坐标从像素坐标转换为归一化坐标（范围在0到1之间）。这是通过将y坐标除以图像高度，x坐标除以图像宽度来实现的。
    # 然后，使用 tf.concat 将这些归一化坐标连接回一个边界框张量。
    new_bbox=tf.concat([tf.cast(ymin,tf.float32)/height,tf.cast(xmin,tf.float32)/weight,tf.cast(ymax,tf.float32)/height,tf.cast(xmax,tf.float32)/weight],2)
    #使用 tf.image.draw_bounding_boxes 函数在原始图像上绘制归一化的边界框，得到新的图像
    new_image=tf.image.draw_bounding_boxes(image,new_bbox)
    #使用 tf.summary.image 函数将绘制了边界框的新图像添加到TensorBoard的日志中，以便进行可视化。
    # 这里的 'input' 是日志中图像的标签。
    tf.summary.image('input',new_image)

#定义函数，用于计算平均精度（Average Precision, AP）的Python函数，平均精度是计算机视觉领域中评估目标检测模型性能的常用指标。
# 这个函数接受两个参数：rec（召回率）和prec（精确率），并计算平均精度AP。
def voc_ap(rec,prec):
    #在召回率列表的开始和结束分别插入0.0和1.0
    rec.insert(0,0.0)
    rec.append(1.0)
    #创建召回率列表的一个副本，以便在后续计算中使用。
    mrec=rec[:]
    #在精确率列表的开始插入0.0，并在结束插入0.0
    prec.insert(0,0.0)
    prec.append(0.0)
    mpre=prec[:]

    #从精确率列表的倒数第二个元素开始向前遍历，确保每个元素不小于其后一个元素。这是为了确保精确率是单调递增的。
    for i in range(len(mpre)-2,-1,-1):
        mpre[i]=max(mpre[i],mpre[i+1])
    i_list=[] #用于存储召回率变化的位置

    #遍历召回率列表，找出召回率发生变化的位置，并存储在i_list中
    for i in range(1,len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    ap=0.0
    #遍历i_list中的索引，计算平均精度。对于每个召回率变化的点，计算该点的精确率与召回率变化的乘积，并累加到ap中。
    for i in i_list:
        ap+=((mrec[i]-mrec[i-1])*mpre[i])
    return ap,mrec,mpre

