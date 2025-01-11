# 双线性插值需要遍历通道channnel，最临近插值不遍历
import cv2
import numpy as np

def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dts_w", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:      # 说明图像没有做任何的缩放
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w,3), dtype=np.uint8)
    scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h      # 缩放比例
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x-0.5         # 使中心重合
                src_y = (dst_y + 0.5) * scale_y-0.5

                # 计算x0,x1,y0,y1
                src_x0 = int(np.floor(src_x))    #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0+1, src_w-1)  # 处理边界值
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1, src_h-1)

                # 计算目标像素值
                temp0 = (src_x1 - src_x)* img[src_y0, src_x0, i] +(src_x-src_x0)*img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x)* img[src_y1, src_x0,i] + (src_x-src_x0)*img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1-src_y)*temp0 + (src_y-src_y0)*temp1)

    return dst_img

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img,(700, 700))
    cv2.imshow("bilinear interp", dst)
    cv2.waitKey(0)
