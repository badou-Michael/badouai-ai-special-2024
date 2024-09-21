import cv2
import numpy as np
from matplotlib import pyplot as plt
#最邻近插值
#定义一个最邻近插值函数
def function(photo):
    h,w,cha=photo.shape
    emptyImage=np.zeros((800,800,cha),np.uint8)
    sh=800/h
    sw=800/w
    for i in range(800):
        for j in range(800):
            # 计算原图像中的坐标，并进行边界检查
            x = min(int(np.floor(i / sh)), h - 1)
            y = min(int(np.floor(j / sw)), w - 1)
            # 从原图像中复制像素到新图像的对应位置
            emptyImage[i, j] = photo[x, y]
    return emptyImage
#双线性插值
def bilinear_interpolation(photo,input_dim):
    h,w,cha=photo.shape
    out_h=input_dim[0]
    out_w=input_dim[1]
    if h == out_h and w == out_w:
        return photo.copy()
    out_photo=np.zeros((out_h,out_w,3),dtype=np.uint8)
    scale_x, scale_y = float(w) / out_w, float(h) / out_h
    for i in range(cha):
        for dst_y in range(out_h):
            for dst_x in range(out_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5


                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * photo[src_y0, src_x0, i] + (src_x - src_x0) * photo[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * photo[src_y1, src_x0, i] + (src_x - src_x0) * photo[src_y1, src_x1, i]
                out_photo[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return out_photo
"""
证明中心重合+0.5:
由题可得：因为：Xn=(N/M)*Xm => Xm=Xn*(m/n)
         则：存在中心几何点的关系为：X((m-1)/2)=X((n-1)/2)*(m/n)
         设存在Z使得：(m-1)/2 + Z=(m/n)*((n-1)/2+Z)
                 =>  (m-1)/2-((m/n)*(n-1))/2=(m/n-1)*Z
                 =>  (m/n-1)/2=(m/n-1)/Z
                 =>  Z=1/2=0.5
"""
if __name__ == '__main__':
    photo=cv2.imread("ho.jpg")
    photo_zoom=function(photo)#调用最邻插值函数
    cv2.imshow("nearest interp",photo_zoom)#显示最邻插值后的图片
    out_photo=bilinear_interpolation(photo,(800,800))
    cv2.imshow("bilinear interp",out_photo)#显示双线性插值后的图片

#实现直方图均衡化
# 获取灰度图像
photo = cv2.imread("ho.jpg", 1)
gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)


# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("灰度",gray)
cv2.imshow("均衡化后",dst)
cv2.waitKey(0)
