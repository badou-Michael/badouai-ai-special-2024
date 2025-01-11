import cv2
import numpy as np
def single_linear(img):
    src_h,src_w,channel=img.shape
    out_img=np.zeros((900,900,channel),np.uint8)
    dst_h,dst_w=900,900
    sw=float(src_w) / dst_w
    sh=float(src_h) / dst_h
    for i in range(channel):
        for dst_x in range(dst_w-1):
            for dst_y in range(dst_h-1):
                src_x = (dst_x + 0.5) * sw - 0.5
                src_y = (dst_y + 0.5) * sh - 0.5
                src_x0=int(np.floor(src_x))
                src_x1=src_x0+1
                src_y0=int(np.floor(src_y))
                src_y1=src_y0+1
                out_img[dst_x,dst_y,i]=img[src_x0,src_y0,i]*(src_x1-src_x)/(src_x1-src_x0)+img[src_x1,src_y1,i]*(src_x-src_x0)/(src_x1-src_x0)
    return out_img
def main():
    img=cv2.imread("lenna.png")
    output_img=single_linear(img)
    cv2.imshow("before",img)
    cv2.imshow("after",output_img)
    cv2.waitKey(0)



if __name__=="__main__":
    main()
