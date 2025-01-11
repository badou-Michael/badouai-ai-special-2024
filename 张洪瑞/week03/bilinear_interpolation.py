'''
双线性插值
'''
import cv2
import numpy as np

def func(img, size):
    src_height, src_width, channel = img.shape[:3]
    dst_height = size[1]
    dst_width = size[0]
    # 原图与目标图之间的比例
    scale_x = float(src_width) / dst_width
    scale_y = float(src_height) / dst_height
    zoom = np.zeros((dst_height, dst_width, channel), dtype=np.uint8)
    for c in range(3):
        for dst_h in range(dst_height):
            for dst_w in range(dst_width):
                # 目标像素点对应的原图像上的点并且进行图像中心点对齐
                src_x = scale_x * dst_w + 0.5 * (scale_x - 1)
                src_y = scale_y * dst_h + 0.5 * (scale_y - 1)

                # 计算保存目标图像素周边的点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_width - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_height - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                zoom[dst_h, dst_w, c] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) *temp1)
    return zoom

pth = "lenna.png"
img = cv2.imread(pth)
zoom = func(img, (800, 800))
print(zoom.shape)
cv2.imshow("Image", img)
cv2.imshow("Bilinear", zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()
