import numpy as np
import cv2

lenna = cv2.imread("./lenna.png")
src_w, src_h = lenna.shape[ : 2]
src_array = np.array(lenna)
dst_w = dst_h = 1000
dst_array = np.zeros((dst_w, dst_h, src_array.shape[2]), dtype=np.uint8)
scale_x = scale_y = src_w / dst_w

for channel in range(src_array.shape[2]):
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            # 几何中心对齐
            src_x = (dst_x + 0.5) * scale_x - 0.5
            src_y = (dst_y + 0.5) * scale_y - 0.5
            # src_x = dst_x * scale_x
            # src_y = dst_y * scale_y

            src_x0 = int(src_x)
            src_y0 = int(src_y)
            src_x1 = min(src_x0 + 1, src_w - 1)
            src_y1 = min(src_y0 + 1, src_h - 1)

            # 在x方向进行单线性插值
            tmp0 = (src_x1 - src_x) * src_array[src_x0, src_y0, channel] + (src_x - src_x0) * src_array[src_x1, src_y0, channel]
            tmp1 = (src_x1 - src_x) * src_array[src_x0, src_y1,channel] + (src_x - src_x0) * src_array[src_x1, src_y1, channel]

            # 在y方向进行单线性插值并赋值给变换后的图像
            dst_array[dst_x, dst_y, channel] = int((src_y1 - src_y) * tmp0 + (src_y - src_y0) * tmp1)

cv2.imwrite("./biglenna.png", dst_array)
cv2.imshow("biglenna", dst_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
