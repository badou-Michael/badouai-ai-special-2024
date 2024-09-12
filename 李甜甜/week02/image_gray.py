
import cv2
import numpy as np
img = cv2.imread("lenna.png")  #读取图像数据
print(img)
h,w = img.shape[:2]  #.shape cv2带的函数 会返回元组包含三个值，高宽通道数，只取前两个高宽
#print(h,w)
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = m[0]*0.11 + m[1]*0.59 + m[2]*0.3
#cv2 是以gbr的顺序h
#print(img_gray)
print("img show gray%s"%img_gray)
cv2.imshow("img show gray",img_gray)



#二值图
h,w = img_gray.shape[:2]   #获取高宽
#建立一个全为零的，跟img数据类型相同的，高为h，宽为w 图片数据
img_bw =np.zeros([h,w],img.dtype)
#先把矩阵除以255
img_gray = img_gray/255
#接下来根据img_gray每个像素点的数值，给给img_gray赋值
for i in range(h):
    for j in range(w):
        if img_gray[i,j]>0.5:
            img_bw[i,j] = 1
        else:
            img_bw[i,j] = 0
img_bw = img_bw *255
print(img_bw)
#根据数据显示图片
cv2.imshow("black_withe_img",img_bw)

# 添加等待按键操作
cv2.waitKey(0)

# 销毁窗口
cv2.destroyAllWindows()
