import cv2
import numpy as np

#临近值插值
def function(img, out_dim):
    h,w,channels = img.shape
    d_w,d_h=out_dim[0],out_dim[1]
    emptyImage = np.zeros((d_w,d_h,channels),np.uint8)
    sh=d_w/h
    sw=d_h/w
    for i in range(d_h):
        for j in range(d_w):
            x=int(i/sh+0.5)  #int(),转为整型，向下取整
            y=int(j/sw+0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage

#双线性插值
def function_bin(img,out_dim):
    src_h,src_w,channel = img.shape
    dst_w,dst_h=out_dim[0],out_dim[1]
    if(src_h==dst_h and src_w==dst_w):
        return img.copy()
    dst_img = np.zeros((dst_w,dst_h,3),dtype=np.uint8)#创建目标图像 三通套
    scale_x,scale_y=float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #计算中心位置
                src_x = (dst_x+0.5)*scale_x-0.5
                src_y = (dst_y+0.5)*scale_y-0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w -1)#防止坐标越界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1,src_h -1)#防止坐标越界
                #计算插值 通过双线性公式
                temp0=(src_x1-src_x)*img[src_x0,src_y0,i]+(src_x-src_x0)*img[src_x1,src_y0,i]
                temp1=(src_x1-src_x)*img[src_x0,src_y1,i]+(src_x-src_x0)*img[src_x1,src_y1,i]
                #p
                dst_img[dst_x,dst_y,i]=int((src_y1-src_y)*temp0+(src_y-src_y0)*temp1)

    return dst_img





if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # cv2.resize(img.(800,800,c),near/bin)
    zoom = function(img,(800,800))

    bin_image = cv2.resize(img,(800,800), interpolation=cv2.INTER_LINEAR)
    dst_img = function_bin(img,(800,800))
    print(zoom)
    print(zoom.shape)
    cv2.imshow("image", img)
    cv2.imshow("fun_nearest_interp", zoom)
    cv2.imshow("cv_bin_image",bin_image)
    cv2.imshow("fun_bin_image",dst_img)
    cv2.waitKey(0)
