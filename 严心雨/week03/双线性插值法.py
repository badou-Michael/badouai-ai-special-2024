# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import cv2

def billinear_interpolation(src_image,des_image):
    scr_h,scr_w,scr_c=scr_image.shape
    des_h, des_w = des_image[:2]
    #des_h,des_w=des_image[0],des_image[1] #不能用 des_image[:2] 因为des_image参数输入时是元组形式，而元组不能被修改
    print("scr_h,scr_w:",scr_h,scr_w)#“后要加逗号
    print("des_h,des_w:",des_h,des_w)

    if scr_h == des_h and scr_w == des_w:#判断两者是否相等，要用==，=代表赋值
        return scr_iamge.copy()
    else:
        out_image = np.zeros((des_h, des_w, scr_c), dtype=np.uint8)#建表的大小要建目标表的大小，不能建成和原表一样的大小
        scale_y=scr_h/des_h
        scale_x=scr_w/des_h
        for c in range(scr_c):#通道数 进行channel循环，计算会更准确。P点不直接等于邻近点的像素值，需要计算
            for des_y in range(des_h):#坐标
                for des_x in range(des_w):
                    scr_y = (des_y + 0.5) * scale_y - 0.5 #几何中心对齐
                    scr_x = (des_x + 0.5) * scale_x - 0.5 #确认目标图中的点在原图中的坐标

                    #边界处理
                    scr_x0 = int(scr_x)#向下取整 图像左边的位置
                    scr_x1 = min(scr_x0 + 1,scr_w - 1)#边界处理 图像右边的位置不能超出界限
                    scr_y0 = int(scr_y)
                    scr_y1 = min(scr_y0 + 1,scr_h - 1)

                    #计算像素值
                    #在X方向做插值
                    f1=(scr_x1-scr_x)*scr_image[scr_y0,scr_x0,c]+(scr_x-scr_x0)*scr_image[scr_y0,scr_x1,c]#[高，宽，通道数]
                    f2=(scr_x1-scr_x)*scr_image[scr_y1,scr_x0,c]+(scr_x-scr_x0)*scr_image[scr_y1,scr_x1,c]
                    #在y方向做插值
                    out_image[des_y,des_x,c]=int((scr_y1-scr_y)*f1+(scr_y-scr_y0)*f2) #表示在该（des_y,des_x）点的像素值
        return out_image

if __name__ == '__main__':
    scr_image = cv2.imread('lenna.png')
    des_image = billinear_interpolation(scr_image,(900,900,3))
    cv2.imshow("show des_image",des_image)
    cv2.waitKey(0)
