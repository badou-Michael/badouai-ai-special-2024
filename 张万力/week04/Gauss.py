import cv2
import random
'''
高斯噪声，增加图像噪声
1.选取需要增加噪声的数量
2.循环随机取图像
3.对X，Y进行高斯赋值
4.限制最大最小值
'''
# mu (μ) ,  sigma (σ)
def gauss_noise(img,mu,sigma,percetage):
    img1=img
    num=int(percetage*img.shape[0]*img.shape[1])
    for i in range(num):
        X=random.randint(0,img.shape[0]-1)
        Y=random.randint(0,img.shape[1]-1)
        img1[X,Y]=img1[X,Y]+random.gauss(mu,sigma)
        if img1[X,Y] <0:
            img1[X, Y]=0
        elif img1[X,Y] >255:
            img1[X, Y]=255
    return img1;




img = cv2.imread("../lenna.png",0)  # step1 读取图片
img1 = gauss_noise(img,2,40,1)
img = cv2.imread("../lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


cv2.imshow("gauss image",img1) #输出高斯噪声
cv2.imshow(" image",img2) #输出原图
cv2.waitKey(0)
cv2.destroyAllWindows() #关闭所有openCV创建的窗口
