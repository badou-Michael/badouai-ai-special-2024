import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

if __name__=='__main__':
    
    #1、灰度化
    image=cv2.imread('lena.png')
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #2、高斯平滑
    sigma=0.5 #高斯核参数，标准差，可调
    dim=5 #高斯核尺寸，5*5
    Gaussian_filter=np.zeros([dim,dim]) #设置5*5的0矩阵，用于存储高斯核数据
    n1=1/(2*math.pi*sigma**2) #计算1/2πσ^2 
    n2=-1/(2*sigma**2) #计算-1/2σ^2
    tmp=[i-dim//2 for i in range(dim)] #做中心对称
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i,j]=n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter=Gaussian_filter/Gaussian_filter.sum()
    dx,dy=img.shape
    img_new=np.zeros(img.shape) #用于存储平滑后的图像
    tmp=dim//2
    img_pad=np.pad(img,((tmp,tmp),(tmp,tmp)),'constant') #加padding
    for i in range(dx):
        for j in range(dy):
            img_new[i,j]=np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter) #添加高斯滤波
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8),cmap='gray')
    plt.axis('off')
    
    #3、sobel算子提取图像边缘
    sobel_x=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sx
    sobel_y=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) #sy
    img_tidu_x=np.zeros(img_new.shape) #设置0矩阵存储梯度图像
    img_tidu_y=np.zeros([dx,dy])
    img_tidu=np.zeros(img_new.shape)
    img_pad=np.pad(img_new,((1,1),(1,1)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sobel_x) #x方向
            img_tidu_y[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sobel_y) #y方向
            img_tidu[i,j]=np.sqrt(img_tidu_x[i,j]**2+img_tidu_y[i,j]**2) #梯度图像
    img_tidu_x[img_tidu_x==0]=0.0000001 #如果x方向为0，变为一个极小值
    tan=img_tidu_y/img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8),cmap='gray')
    plt.axis('off')
    
    #4、非极大值抑制
    img_yizhi=np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            temp = img_tidu[i-1:i+2, j-1:j+2] # 获取当前像素点的8邻域矩阵
        # 根据梯度方向判断是否需要抑制
            if (tan[i, j] < -45) or (tan[i, j] >= 45 and tan[i, j] < 135):
                if img_tidu[i, j] >= max(temp[0, 1], temp[1, 0], temp[2, 1]):
                    img_yizhi[i, j] = img_tidu[i, j]
            elif (tan[i, j] >= -45 and tan[i, j] < 0) or (tan[i, j] >= 135 and tan[i, j] < 180):
                if img_tidu[i, j] >= max(temp[0, 1], temp[1, 2], temp[1, 0]):
                    img_yizhi[i, j] = img_tidu[i, j]
            else:
                if img_tidu[i, j] >= max(temp[0, 1], temp[1, 0], temp[1, 2]):
                    img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    #4、双阈值检测、连接边缘
    # 设置高低阈值
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    # 标记“一定是边缘”和“一定不是边缘”的点
    img_yizhi[img_yizhi >= high_boundary] = 255
    img_yizhi[img_yizhi <= lower_boundary] = 0
    
    # 边缘连接
    def connect_edges(img, lower, high):
        stack = np.argwhere(img == 255).tolist()  # 将“一定是边缘”的点添加到栈中
        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    if img[nx, ny] > lower and img[nx, ny] < high:
                        img[nx, ny] = 255
                        stack.append([nx, ny])
        return img
    # 连接边缘
    img_yizhi = connect_edges(img_yizhi, lower_boundary, high_boundary)
    # 清理未连接的“有可能是边缘”的点
    img_yizhi[(img_yizhi > lower_boundary) & (img_yizhi < high_boundary)] = 0
  
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()
