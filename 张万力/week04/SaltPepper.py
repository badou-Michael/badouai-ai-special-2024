import cv2
import random
'''
椒盐噪声，增加图像噪声，0或255
1.选取需要增加噪声的数量
2.循环随机取图像
3.对X，Y进行高斯赋值
4.随机赋值黑或白
'''
# mu (μ) ,  sigma (σ)
def slat_pepper(img,percetage):
    img1=img
    num=int(percetage*img.shape[0]*img.shape[1])
    for i in range(num):
        X=random.randint(0,img.shape[0]-1)
        Y=random.randint(0,img.shape[1]-1)

        if random.random() <0.5:
            img1[X, Y]=0
        else:
            img1[X, Y]=255
    return img1;

img = cv2.imread("../lenna.png",0)  # step1 读取图片
img1 = slat_pepper(img,0.2)
img = cv2.imread("../lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


cv2.imshow("salt pepper image",img1) #输出椒盐噪声
cv2.imshow(" image",img2) #输出灰度
cv2.waitKey(0)
cv2.destroyAllWindows() #关闭所有openCV创建的窗口
