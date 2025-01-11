import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


#def grayscale(path):
#    grayimg=cv2.imread(path,0) 
#    return grayimg

def grayscale(path):
    img = plt.imread(path)
    if path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)
    return img

def gausskel(size, sigma):
    kernel=np.zeros((size,size))
    temp=[i-size//2 for i in range(size)]
    factor=1/(2*np.pi*sigma*sigma)
    for i in range(size):
        for j in range(size):
            kernel[i,j]=factor * np.exp(-(temp[i]**2 + temp[j]**2)/(2 * sigma ** 2))
    kernel/=np.sum(kernel)
    return kernel                   

def blur_gau(kernel,img):
    dx,dy=img.shape
    img_blur=np.zeros((dx,dy))
    pad=kernel.shape[0]//2
    img_pad=np.pad(img,((pad,pad),(pad,pad)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_blur[i,j]=np.sum(img_pad[i:i+kernel.shape[0],j:j+kernel.shape[0]]*kernel)
    plt.title("blur_gau")
    plt.imshow(img_blur.astype(np.uint8),cmap='gray')
    plt.show()
    return img_blur

#using convolve2d function

def GxGy(img):
    Gx=np.array([[-1, 0, 1], 
                 [-2, 0, 2], 
                 [-1, 0, 1]])
    Gy=np.array([[1, 2, 1], 
                 [ 0,  0,  0], 
                 [ -1,  -2,  -1]])
    img_x=convolve2d(img,Gx,mode='same',boundary='fill',fillvalue=0)
    img_y=convolve2d(img,Gy,mode='same',boundary='fill',fillvalue=0)
    img_sobel=np.sqrt(img_x**2+img_y**2)
    #print(img_sobel)
    #img_sobel=(img_sobel/img_sobel.max())*255
    plt.title("img_sobel")
    plt.imshow(img_sobel.astype(np.uint8),cmap="gray")
    plt.show()
    return img_sobel,img_x, img_y

def nonmax(img_sobel,img_x,img_y):
    angle=img_y/img_x
    angle[angle==0]=0.00000001
    img_tidu=img_sobel
    img_nonmax = np.zeros(img_tidu.shape)
    for i in range(1, img_nonmax.shape[0]-1):
        for j in range(1, img_nonmax.shape[1]-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_nonmax[i, j] = img_tidu[i, j]
    plt.title("non max yizhi")
    plt.imshow(img_nonmax.astype(np.uint8), cmap='gray')
    plt.show()
    return img_nonmax

def threshold(low,ratio,img_nonmax):
    high=low*ratio
    strong=[]
    for i in range(1,img_nonmax.shape[0]-1):
        for j in range(1,img_nonmax.shape[1]-1):
            if img_nonmax[i,j]>=high:
                img_nonmax[i,j]=255
                strong.append([i,j])
            elif img_nonmax[i,j]<=low:
                img_nonmax[i,j]=0
    #中间值
    def check(high,low,i,j,img_nonmax,strong):
        if low<=img_nonmax[i,j]<=high:
            img_nonmax[i,j]=255
            strong.append([i,j])
    
    while len(strong)>0:
        temp1,temp2=strong.pop()
        for a in [-1,0,1]:
            for b in [-1,0,1]:
                if a==0 and b==0:
                    continue
                check(high,low,temp1+a,temp2+b,img_nonmax,strong)
            
    for i in range(img_nonmax.shape[0]):
        for j in range(img_nonmax.shape[1]):
            if img_nonmax[i, j] != 0 and img_nonmax[i, j] != 255:
                img_nonmax[i, j] = 0
     

    plt.title("canny")
    plt.imshow(img_nonmax.astype(np.uint8),cmap="gray")
    plt.show()
    

def canny(path,size,sigma):
    img=grayscale(path)
    kernel=gausskel(size,sigma)
    img_blur=blur_gau(kernel,img)
    img_sobel, img_x, img_y = GxGy(img_blur)
    img_nonmax=nonmax(img_sobel,img_x,img_y)
    low=img_sobel.mean()*0.5
    threshold(low,3,img_nonmax)
    
canny('lenna.png',5,0.5)
