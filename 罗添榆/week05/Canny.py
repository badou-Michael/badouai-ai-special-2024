from PIL import Image
from matplotlib import pyplot as plt
import math
from PIL import ImageFilter

im = Image.open("lenna.png")
im_gray = im.convert('L')   #rgb转灰度图
im_gray.show()  #显示灰度图

#高斯平滑
im_gray = im_gray.filter(ImageFilter.SMOOTH)

#Image转为list类型
im_array = im_gray.load()
im_list = [[0 for i in range(im.size[0])] for j in range(im.size[1])]
for i in range(im.size[1]):
    for j in range(im.size[0]):
        im_list[i][j] = im_array[j, i]
plt.figure("灰度图")
plt.imshow(im_list, cmap='gray')
#plt.show()

#求梯度 （梯度模值和方向）
#方向：  0：0度(x)    1: 45度   2： 90度（y） 3: 135度
grad = [[[0, 0] for i in range(im.size[0])] for j in range(im.size[1])]  #170 * 200 * 2 每个像素点的梯度大小和方向
for j in range(1, im.size[1] - 1):
    for i in range(1, im.size[0] - 1):
        #x方向梯度
        grad_x = im_list[j + 1][i + 1] + im_list[j - 1][i + 1] + 2 * im_list[j][i + 1] - \
                 im_list[j - 1][i - 1] - im_list[j + 1][i - 1] - 2 * im_list[j][i - 1]
        #y方向梯度
        grad_y = im_list[j + 1][i - 1] + im_list[j + 1][i + 1] + 2 * im_list[j + 1][i] - \
                 im_list[j - 1][i - 1] - im_list[j - 1][i + 1] - 2 * im_list[j - 1][i]
        grad_x = math.floor(grad_x / 4)
        grad_y = math.floor(grad_y / 4)
        #合梯度
        grad[j][i][0] = math.floor(math.sqrt(grad_x * grad_x + grad_y * grad_y))
        if(grad[j][i][0] > 255):
            grad[j][i][0] = 255
        #if(grad[j][i][0] < 50):
            #grad[j][i][0] = 0
        #梯度方向
        if(grad_x == 0):
            grad[j][i][1] = 2  #y方向
        else:
            theta = math.atan2(grad_y, grad_x)
            if(math.fabs(theta) < math.pi / 8):
                grad[j][i][1] = 0 #x方向
            elif(theta > 0):
                if(math.fabs(theta) <math.pi * 3 / 8):
                    grad[j][i][1] = 1  #45度方向
                else:
                    grad[j][i][1] = 2#y方向
            else:
                if (math.fabs(theta) < math.pi * 3 / 8):
                    grad[j][i][1] = 3  # 135度方向
                else:
                    grad[j][i][1] = 2  # y方向

#显示梯度图
img = [[0 for i in range(im.size[0])] for j in range(im.size[1])]
for i in range(im.size[1]):
    for j in range(im.size[0]):
        img[i][j] = grad[i][j][0]
plt.figure("梯度图")
plt.imshow(img, cmap='gray')
# plt.show()

#非极大值抑制,使边缘更清晰和细
img2 = img
for j in range(1, im.size[1] - 1):
    for i in range(1, im.size[0] - 1):
        dir = grad[j][i][1]
        grad_now = grad[j][i][0]
        if (dir == 0):  #梯度方向为x
            if(grad_now < grad[j][i + 1][0] or grad_now < grad[j][i - 1][0]):
                # grad[j][i][0] == 0
                img2[j][i] = 0
                print('0')
        elif(dir == 1):#45度方向
            if(grad_now < grad[j + 1][i + 1][0] or grad_now < grad[j - 1][i - 1][0]):
                # grad[j][i][0] == 0
                img2[j][i] = 0
                print('1')
        elif(dir == 2): #y方向
            if(grad_now < grad[j + 1][i][0] or grad_now < grad[j - 1][i][0]):
                # grad[j][i][0] == 0
                img2[j][i] = 0
                print('2')
        else:  #145度
            if(grad_now < grad[j + 1][i - 1][0] or grad_now < grad[j - 1][i + 1][0]):
                # grad[j][i][0] == 0
                img2[j][i] = 0
                print('3')

#显示非极大值抑制后的图像
plt.figure("非极大值抑制图")
plt.imshow(img2, cmap='gray')
# plt.show()

#双阈值检测
low = 10
high = 30

for i in range(im.size[1]):
    for j in range(im.size[0]):
        if(img2[i][j] < low):
            img2[i][j] = 0
        elif(img2[i][j] > high):
            img2[i][j] = 255

#双阈值检测后的图像
plt.figure("双阈值检测图")
plt.imshow(img2, cmap='gray')
# plt.show()

#抑制孤立的弱边缘
listx = [-1, 0, 1]
listy = [-1, 0, 1]
img3 = img2
for j in range(1, im.size[1] - 1):
    for i in range(1, im.size[0] - 1):
        flag = 1
        for dx in range(len(listx)):
            for dy in range(len(listy)):
                j = j + listy[dy]
                i = i + listx[dx]
                if(img2[j][i] == 255):
                    flag = 0
                    break
            if(not flag):
                break
        if(flag):
            img3[j][i] = 0

#抑制孤立的弱边缘后的图像
# print('ssssssss')
plt.figure("孤立弱边缘图")
plt.imshow(img3, cmap='gray')
plt.show()
