import cv2
import numpy as np

#--------------------------------手动计算临近插值算法---------------------------
def function(image):
    h,w,t = image.shape         #提取输入图片的参数
    dst_x = 1000/h            #计算出拓展图片和原图的尺寸比例
    dst_y = 1000/w
    new_image = np.zeros((1000,1000,t),np.uint8)      #创建新的空图片
    for i in range(1000):
        for j in range(1000):
            x = int( i / dst_x+0.5)         #遍历原图每个像素的 和 尺寸比例相乘的到拓展的点位位置  加0.5是为了四舍五入
            y = int( j / dst_y+0.5)          #一定要转int
            new_image[i,j] = image[x,y]

    return  new_image

main_image = cv2.imread("lenna.png")
img = function(main_image)
cv2.imshow("窗口1",img)
cv2.waitKey()


#openCV 自带的临近插值算法
main_image = cv2.resize(main_image,(1000,1000),interpolation=cv2.INTER_NEAREST)    #双线性插值：cv2.INTER_LINEAR
cv2.imshow("窗口2",main_image)
cv2.waitKey()
