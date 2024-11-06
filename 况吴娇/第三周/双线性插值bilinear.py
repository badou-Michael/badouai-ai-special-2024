import cv2
import numpy as np
print("双线性插值")

def bilinear_interpolation (img,out_dim):
  src_h,src_w,channels=img.shape
  dst_h,dst_w=out_dim[0],out_dim[1]
  print("源高,源宽 ，通道数= ", src_h, src_w,channels)
  print("目标高, 目标宽 = ", dst_h, dst_w)
  if src_h==dst_h and src_w==dst_w:
      return img.copy()
  dst_img=np.zeros((dst_h,dst_w,channels),dtype=np.uint8)
  print('目标图高宽通道数',dst_img.shape)
  ##求目标图 在原图的坐标
  scale_x,scale_y=float(src_h)/dst_h,float(src_w)/dst_w
  for i in range(channels):
      for dst_y in range (dst_h):
          for dst_x in  range(dst_w):
#中心对称，提高数据质量。目标图 在原图的坐标
# 加上0.5是为了将坐标从以左上角为原点的坐标系
# （图像坐标通常以左上角为原点，x轴向右增加，y轴向下增加）转换为以像素中心为参考的坐标系。
# 这样做是因为在双线性插值中，我们通常希望以像素的中心点为基准进行计算。
             src_x=(dst_x+0.5)*scale_x-0.5
             src_y=(dst_y+0.5)*scale_y-0.5
#确定用于计算插值的四个最近邻像素的四个像素点，它们形成一个矩形区域
             src_x0=int(np.floor(src_x))  #np.floor()返回不大于输入参数的最大整数。（向下取整）
             src_x1=min(src_x0+1,src_w-1)  #in() 函数确保 src_x1 不会超出图像的宽度。src_w - 1 是图像宽度的最大有效索引。
             src_y0=int(np.floor(src_y))
             src_y1=min(src_y0+1,src_h-1) #min() 函数确保 src_y1 不会超出图像的高度。src_h - 1 是图像高度的最大有效索引。
             #x方向插值
             tmp0=(src_x1-src_x)*img[src_y0,src_x0,i]   +(src_x-src_x0)* img[src_y0,src_x1,i]
             tmp1=(src_x1-src_x)*img[src_y1,src_x0,i]   +(src_x-src_x0)* img[src_y1,src_x1,i]
            #y轴
             dst_img[dst_y,dst_x,i]=int(tmp0*(src_y1-src_y )+ tmp1*(src_y-src_y0))
          #int() 函数在这里用于将计算结果转换为整数。这是必要的，因为图像的像素值通常表示为整数。
          # 在大多数图像处理库（如OpenCV）中，图像数据是以整数形式存储的，每个像素的值通常在0到255之间（对于8位图像）。
  return  dst_img


if __name__=='__main__':
    img=cv2.imread("lenna.png")##彩图
    dst=bilinear_interpolation(img,(700,700))##扩大
    cv2.imshow('bilinear_interpolation',dst)
    cv2.waitKey(0)
