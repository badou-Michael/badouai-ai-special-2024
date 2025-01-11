'''
canny 边缘检测算法，分为五个步骤
1. 实现灰度化，为了减少计算量


'''
import math

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    '''
    1.图像灰度化
    '''
    img = plt.imread('../images/lenna.png')
    # plt 模块读取png格式图像以浮点形式 ，需要转换【0-255】
    img = img * 255
    #算术平均化(R+G+B/3) 而非加权平均(图像的灰度化gray)
    img = img.mean(axis=-1)
    print("image------------gray")
    print(img)


    '''
    2.高斯平滑
    高斯滤波是非常好的降噪滤波 ，使得像素值符合高斯分布
    为什么这里要用高斯滤波，目的为了降噪 ，是一种优化项
    边缘检测是局部区域像素值变化大，是一种高频信号，所以图像中
    如果有高频噪声，会影响检测结果
    2.1  构造5x5 gaussian_kernel 根据数学公式求出
    2.2  hxw图像与G_kernel（5x5）做卷积，加权求和
    '''

    sigma = 0.5
    G_kernel_size = 5
    G_kernel = np.zeros([G_kernel_size,G_kernel_size])

    #kernel 坐标，中心为（0，0），5x5 (-2，-1,0,1,2）
    temp = [i-G_kernel_size//2 for i in range(G_kernel_size)]
    print('temp =  ',temp)

    #公式
    n1 = 1/2*math.pi*sigma**2
    n2 = -1/2**sigma

    for i in range(G_kernel_size):
        for j in range(G_kernel_size):
            G_kernel[i,j]=n1*math.exp(n2*(temp[i]**2+temp[j]**2)) #有些值可能会超出【0-255】，要归一化
            #每一个元素除所有元素之和
    print('非归一化的G_kernel +++++++++++++++++++++\n', G_kernel)
    G_kernel = G_kernel/np.sum(G_kernel)  # 归一化
    print('归一化的G_kernel +++++++++++++++++++++\n',G_kernel)




    #做padding,保证矩阵的最外围被卷积到
    #d = f-1/2
    d = G_kernel_size-1//2
    img_padding = np.pad(img,(d,d),'constant')
    dx,dy =img.shape
    img_new = img
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] =np.sum(img_padding[i:i+G_kernel_size,j:j+G_kernel_size]*G_kernel)





    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')




    '''
    3.使用sobel算子求梯度  ，
    构建sobel 算子，Gx = ,Gy=
    '''

    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    img_tudu_x = np.zeros(img_new.shape)
    img_tudu_y = np.zeros(img_new.shape)
    img_tudu = np.zeros(img_new.shape)
    # 3x3的sobel 卷积核 ，加一圈padding
    sobel_kernel_size = sobel_kernel_x.shape[0]
    print('sobel_kernel_size=',sobel_kernel_size)
    sobel_d = sobel_kernel_size-1//2  #d = f-1/2
    img_new_tudu  = np.pad(img_new,(sobel_d,sobel_d),'constant')

    for i in range(dx):
        for j in range(dy):
            img_tudu_x[i,j] = np.sum(img_new_tudu[i:i+sobel_kernel_size,j:j+sobel_kernel_size]*sobel_kernel_x)
            img_tudu_y[i,j] = np.sum(img_new_tudu[i:i+sobel_kernel_size,j:j+sobel_kernel_size]*sobel_kernel_y)
            img_tudu[i,j] =np.sqrt(img_tudu_x[i,j]**2 + img_tudu_y[i,j]**2)
    img_tudu_x[img_tudu_x == 0] = 0.00000001
    angle = img_tudu_y/img_tudu_x
    print('angle梯度：\n',angle)
    plt.figure(2)
    plt.imshow(img_tudu.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')



    '''
    4. 非极大抑制 ： 
    这里卡了好久，梯度方向，num1和num2求法不一样，canny论文中提到gA=w*g1+(1-w)*g2,gB=w*g3+(1-w)*g4
    https://zhuanlan.zhihu.com/p/600529269?utm_psn=1834417971394654208
    '''
    img_yuzhi = np.zeros(img_tudu.shape)   #全0矩阵
    for i in range(1,dx-1):
        for j in range(1,dy-1):
            # 提取（i,j）梯度幅值的8领域矩阵
            temp = img_tudu[i-1:i+2,j-1:j+2]
            # 标识变量 flge = true   默认该点非极大值
            flge = True
            '''
            第一种情况 ，G_y>G_x ，角度量为负， G_x 为负且 G_y 为正，则方向为135 ，位于第二象限  w = G_x/G_y
            第二种情况：就是老师在代码中，运行结果是效果一致的
            '''
            if(angle[i,j]<=-1):
                num1 = (temp[0,1]-temp[0,0])/angle[i,j]+temp[0,1]
                num2 = (temp[2,1]-temp[2,2])/angle[i,j]+temp[2,1]
                # w = angle[i,j]
                # g1 = temp[0,2]
                # g3 = temp[2,0]
                # g2 = temp[0,1]
                # g4 = temp[2,1]
                # num1 = w * g1+(1-w)*g2
                # num2 = w * g3+(1-w)*g4
                if(img_tudu[i,j]>num1 and  img_tudu[i,j]>num2):
                    flge = False
            elif(angle[i,j]>=1):
                num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                # w = angle[i, j]
                # g1 = temp[0, 0]
                # g3 = temp[2, 2]
                # g2 = temp[0, 1]
                # g4 = temp[2, 1]
                # num1 = w * g1 + (1 - w) * g2
                # num2 = w * g3 + (1 - w) * g4
                if (img_tudu[i, j] > num1 and img_tudu[i, j] > num2):
                    flge = False
            elif (angle[i, j] > 0):
                num1 = (temp[0, 2] - temp[0, 2]) * angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                # w = angle[i, j]
                # g1 = temp[2,0]
                # g3 = temp[0, 2]
                # g2 = temp[1, 0]
                # g4 = temp[1 ,2]
                # num1 = w * g1 + (1 - w) * g2
                # num2 = w * g3 + (1 - w) * g4
                if (img_tudu[i, j] > num1 and img_tudu[i, j] > num2):
                    flge = False
            elif (angle[i, j] < 0):
                num1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                # w = angle[i, j]
                # g1 = temp[0, 0]
                # g3 = temp[2, 2]
                # g2 = temp[1, 0]
                # g4 = temp[1, 2]
                # num1 = w * g1 + (1 - w) * g2
                # num2 = w * g3 + (1 - w) * g4
                if (img_tudu[i, j] > num1 and img_tudu[i, j] > num2):
                    flge = False
            if flge:
                img_yuzhi[i,j] = img_tudu[i,j]
    print("============================")
    print('img_yuzhi=\n',img_yuzhi)
    plt.figure(3)
    plt.imshow(img_yuzhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    '''
    5.双阈值检测 ：为什么要在非极大抑制后做双阈值检测，图像中可能由噪声或
    其它原因，这种情况下像素点表现为强信号，但它不一定是边缘，称之为假边缘 
    假定给定两个数A  B (A>B 具体要根据需要调整)，如果当前的
    梯度幅值>高阈值        =====>  真实边缘
    高阈值<梯度幅值<低阈值  =====>  弱边缘
    低阈值>梯度幅值        =====>  抑制掉
    
    真实边缘引起的弱边缘像素将连接到强边缘像素，而噪声响应未连接
    查看弱边缘像素及其8个邻域像素，只要其中一个为强边缘像素，则该弱边缘点就可以保留为真实的边缘
    
    '''
    low = img_tudu.mean()*0.5
    high = low*3

    '''
    入栈规则：把真实边缘255入栈 ，然后检测该点的8领域是否有弱边缘，有则为真时边缘，再次入栈，
    一直循环该操作  00 01 02 
                  10    12
                  20 21 22
    '''
    zhan = []

    for i in range(img_yuzhi.shape[0]):
        for j in range(img_yuzhi.shape[1]):
            if img_yuzhi[i,j]>=high:
                img_yuzhi[i,j] = 255
                zhan.append([i,j])
            if img_yuzhi[i,j]<=low:
                img_yuzhi[i,j] = 0

    while len(zhan)>1:

        i, j =  zhan.pop()
        temp = img_yuzhi[i-1:i+2,j-1:j+2]

        if temp[0,0] <= high and temp[0,0] >= low:
            img_yuzhi[i-1,j-1] = 255
            zhan.append([i-1,j-1])
        elif temp[0,1] <= high and temp[0,1] >= low:
            img_yuzhi[i-1,j] = 255
            zhan.append([i-1,j])
        elif temp[0,2] <= high and temp[0,2] >= low:
            img_yuzhi[i-1,j+1] = 255
            zhan.append([i-1,j+1])
        elif temp[1,0] <=high and temp[1,0] >=low:
            img_yuzhi[i,j-1] = 255
            zhan.append([i,j-1])
        elif temp[1, 2] <= high and temp[1, 2] >= low:
            img_yuzhi[i, j+1] = 255
            zhan.append([i, j+1 ])
        elif temp[2,0] <=high and temp[2,0] >=low:
            img_yuzhi[i+1,j-1] = 255
            zhan.append([i+1,j-1])
        elif temp[2,1] <=high and temp[2,1] >=low:
            img_yuzhi[i+1,j] = 255
            zhan.append([i+1,j])
        elif temp[2,2] <=high and temp[2,2] >=low:
            img_yuzhi[i+1,j+1] = 255
            zhan.append([i+1,j+1])

    for i in range(img_yuzhi.shape[0]):
        for j in range(img_yuzhi.shape[1]):
            if img_yuzhi[i,j] !=0 and img_yuzhi[i,j] !=255:
                img_yuzhi[i,j] = 0

        # 绘图]
    plt.figure(4)
    plt.imshow(img_yuzhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值



    plt.show()
