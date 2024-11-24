import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    pic_path = '../../../request/task2/lenna.png'
    img = plt.imread(pic_path) # plt.imread默认输出0到1的浮点数，所以要扩展到255再计算
    # print('img图片的数据为：\n',img)
    if pic_path[-4:] == '.png':
        img = img*255
    # print('乘以255以后得值:\n',img)
    # numpy中mean均值函数,灰度化处理，axis =-1表示沿着最后一个轴，最后一个值为通道
    img = img.mean(axis = -1)
    # print('均值灰度化后的值:\n',img)
    '''
    老师详细推导方便理解，以上代码用下述表示即可
    img = plt.imread('../../../request/task2/lenna.png')即可
    img = (img*255).mean(axis = -1)
    '''
    # figure1:高斯平滑，图片降噪，让图片质量更好
    # 创建一个具有给定标准差sigma和尺寸dim的二维高斯滤波器。
    # 本代码中sigma设置为0.5，图片噪点不多，这通常是一个比较小的值，意味着滤波器会保留更多的图像细节，同时去除一些噪声。
    sigma = 0.5
    dim = 5
    Gaussian_filter = np.zeros([dim,dim])#初始化高斯滤波器
    print(Gaussian_filter.sum())
    print('滤波初始化，创建一个5行5列的二维数组：\n',Gaussian_filter)
    tmp = [i-dim//2 for i in range(dim)] #[-2, -1, 0, 1, 2]，表示到高斯核距离
    n1 = 1/(2*math.pi*sigma**2)#G(x,y)公式中因子
    n2 = -1/(2*sigma**2)#G(x,y)公式中指数部分
    for i in range(dim):
        for j in range (dim):
            Gaussian_filter[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
            # print('数据第%d行%d列：'%(i+1,j+1), Gaussian_filter[i, j],type(Gaussian_filter[i, j]))
    # Gaussian_filter.sum()是滤波器中所有数据进行求和，数据类型为64位浮点数
    print(Gaussian_filter.sum(),type(Gaussian_filter.sum()))
    # Gaussian_filter / Gaussian_filter.sum()后代表的是初始滤波器在进行高斯处理后，每个数据在滤波器的权重
    # 因此权重相加后必然等于1，这步处理高斯滤波归一化
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    print('归一化后权重总和值：\n',Gaussian_filter.sum())#上一行代码的变量赋值
    print('高斯滤波器的值为：\n',Gaussian_filter)
    print('滤波器shape：\n',Gaussian_filter.shape)
    dx,dy = img.shape
    print('img.shape行列数：\n',img.shape)
    img_new = np.zeros(img.shape)
    tmp = dim // 2
    # pad_width 是一个整数或整数元组，表示在每个轴上需要填充的宽度。
    # 如果是元组，它的长度应该与数组的维度相匹配，tem=2
    img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant')#边缘填补，constant默认为0
    print('边缘填充后数据：\n', img_pad)
    print('img_pad.shape行列数：\n',img_pad.shape)
    for i in range(dx):
        for j in range(dy):
            #遍历填补后的边缘数据和权重相乘后的值，
            # img_pad[i:i+dim, j:j+dim]行和列的切片，
            # 从 img_pad 中提取一个以 (i, j) 为中心的 dim x dim 的子区域。这个子区域包含了当前像素及其周围的像素。
            img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter)
    plt.figure(1)
    '''
    在高斯滤波中，卷积核的权重通常被归一化，使得所有权重的和为1。
    这样可以保证滤波后的图像亮度不会由于权重总和不为1而变得过亮或过暗。
    但是，由于浮点数运算和舍入误差，卷积结果可能不会精确地在0到255的范围内。
    将结果转换为uint8可以确保这些值被适当地缩放和截断到有效的像素值范围内。
    '''
    plt.imshow(img_new.astype(np.uint8),cmap = 'gray')
    print('高斯核卷积后数据：\n',img_new.astype(np.uint8))
    print('img_pad.new行列数：\n',img_new.shape)
    plt.axis('off')#关闭图像坐标轴
    '''
    figure2:求梯度。通过高斯平滑后的图像，经过梯度处理，可以基本显示图像灰度变化剧烈的区域
    '''
    # 定义x、y方向sobel核，卷积核已定义，所以不需要像高斯滤波中那样计算tmp距离
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    # 初始化三个数组，用于存储x方向和y方向的梯度以及最终的梯度幅度
    img_tidu_x = np.zeros(img_new.shape)
    img_tidu_y = np.zeros(img_new.shape)
    img_tidu = np.zeros(img_new.shape)
    # 对平滑后图像边缘填充一行一列，对应下方需要遍历3*3矩阵
    img_pad = np.pad(img_new,((1,1),(1,1)),'constant')
    print('img_pad边缘填充为2行2列行列数：\n', img_pad.shape)
    for i in range(dx):
        for j in range(dy):
            # 通过在整个图像上滑动这个3x3的窗口并重复这个过程，可以得到一个梯度幅度图，用于后续的图像分析和处理。
            # np.sum 函数对逐元素乘法后得到的2D数组进行求和，得到一个标量值。表示当前像素点及其周围像素在水平方向上的梯度强度。
            # 这个标量值代表了当前像素点 (i, j) 处的水平梯度强度。
            img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)
            img_tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y)
            #梯度幅度计算开平方，得到在位置 (i, j) 处的梯度向量的幅度。
            img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2+img_tidu_y[i,j]**2)
    print('img_tidu行列数：\n', img_tidu.shape)
    img_tidu_x[img_tidu_x == 0]=0.00000001#处理梯度零值问题，梯度为零的点通常表示图像在这些点上没有边缘，即它们是平坦区域。
    angle = img_tidu_y/img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    '''
    figure3:非极大值抑制
    '''
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            flag = True
            temp = img_tidu[i-1:i+2,j-1:j+2]#梯度幅值的8邻域矩阵，取>=i-1，<=i+
            '''
           梯度方向：angle[i,j] 是梯度的方向，【它指示了边缘相对于水平轴的角度】。
           教材里的那个蓝线就是这个，最大值都在这个蓝线上。【蓝线线条的方向就是梯度方向】
           因为梯度求的就是最大值，变化最快的值。
           num1_,num2，c点就组成了梯度方向上面的三个点，在梯度幅度内进行变动
           '''
            if angle[i,j] <= -1:
                # -tanθ = angle[i,j]
                #num1 = 1/tanθ*temp[0,0]+(1-1/tanθ)*temp[0,1]
                #num2 = 1/tanθ*temp[2,2]+(1-1/tanθ)*temp[2,1]
                num_1 = (temp[0,1]-temp[0,0])/angle[i,j]+temp[0,1]
                num_2 = (temp[2,1]-temp[2,2])/angle[i,j]+temp[2,1]
                # 非极大值判断
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i,j] >= 1:
                # tanθ = angle[i,j]
                # num1 = 1/tanθ*temp[0,2]+(1-1/tanθ)*temp[0,1]
                # num2 = 1/tanθ*temp[2,0]+(1-1/tanθ)*temp[2,1]
                num_1 = (temp[0,2]-temp[0,1])/angle[i,j]+temp[0,1]
                num_2 = (temp[2,0]-temp[2,1])/angle[i,j]+temp[2,1]
                if not (img_tidu[i,j]>num_1 and img_tidu[i,j]>num_2):
                    flag = False
            elif angle[i,j] >0:
                # tanθ = angle[i,j]
                # num1 = tanθ*temp[0,2]+(1-tanθ)*temp[1,2]
                # num2 = tanθ*temp[2,0]+(1-tanθ)*temp[1,0]
                num_1 = (temp[0,2]-temp[1,2])*angle[i,j]+temp[1,2]
                num_2 = (temp[2,0]-temp[1,0])*angle[i,j]+temp[1,0]
                if not(img_tidu[i,j] > num_1 and img_tidu[i,j] > num_2):
                    flag = False
            elif angle[i,j] < 0:
                # -tanθ = angle[i,j]
                # num1 = tanθ*temp[0,0]+(1-tanθ)*temp[1,0]
                # num2 = tanθ*temp[2,2]+(1-tanθ)*temp[1,2]
                num_1 = (temp[1,0]-temp[0,0])*angle[i,j]+temp[1,0]
                num_2 = (temp[1,2]-temp[2,2])*angle[i,j]+temp[1,2]
                if not (img_tidu[i,j]> num_1 and img_tidu[i,j] > num_2 ):
                    flag = False
            if flag:#不符合上面的条件的，那个梯度值就是最大值
                img_yizhi[i,j] = img_tidu[i,j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8),cmap ='gray')
    '''
    figure4:双阈值检测，连接边缘，遍历所有一定是变得点，查看8领域是否存在有可能是边的点，进栈
    '''
    lower_boundary = img_tidu.mean()*0.5
    high_boundary = lower_boundary*3
    zhan = []
    '''
    对非极大值抑制的图形数据进行处理n
    n>=高阈值，标记为强边缘像素
    低阈值 < n < 高阈值，标记为弱边缘
    n<低阈值怎会被抑制
    高弱像素全部标记高亮255值
    '''
    for i in range(1,img_yizhi.shape[0]-1):
        for j in range(1,img_yizhi.shape[1]-1):
            if img_yizhi[i,j] >= high_boundary:
                img_yizhi[i,j] = 255
                zhan.append([i,j])
            elif img_yizhi[i,j] <= lower_boundary:
                img_yizhi[i,j] = 0
    # 抑制孤立低阈值点，区分噪声点与弱边缘点。因为噪声点是孤立的
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()#出栈对应进栈
        a = img_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]#非极大值8领域矩阵
        if (a[0,0] < high_boundary) and (a[0,0] > lower_boundary):
            img_yizhi[temp_1-1,temp_2-1] = 255
            zhan.append([temp_1-1,temp_2-1])
        if (a[0,1] < high_boundary) and (a[0,1] > lower_boundary):
            img_yizhi[temp_1-1,temp_2] = 255
            zhan.append([temp_1-1,temp_2])
        if (a[0,2] < high_boundary) and (a[0,2] > lower_boundary):
            img_yizhi[temp_1-1,temp_2+1] = 255
            zhan.append([temp_1-1,temp_2+1])
        if (a[1,0] < high_boundary) and (a[1,0] > lower_boundary):
            img_yizhi[temp_1,temp_2-1] = 255
            zhan.append([temp_1,temp_2-1])
        if (a[1,2] < high_boundary) and (a[1,2] > lower_boundary):
            img_yizhi[temp_1,temp_2+1] = 255
            zhan.append([temp_1,temp_2+1])
        if (a[2,0] < high_boundary) and (a[2,0] > lower_boundary):
            img_yizhi[temp_1+1,temp_2-1] = 255
            zhan.append([temp_1+1,temp_2-1])
        if (a[2,1] < high_boundary) and (a[2,1] > lower_boundary):
            img_yizhi[temp_1+1,temp_2] = 255
            zhan.append([temp_1+1,temp_2])
        if (a[2,2] < high_boundary) and (a[2,2] > lower_boundary):
            img_yizhi[temp_1+1,temp_2+1] = 255
            zhan.append([temp_1+1,temp_2+1])
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi [i,j] != 0 and img_yizhi[i,j] != 255:
                img_yizhi[i,j] = 0
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap = 'gray')
    plt.show()



