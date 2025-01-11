import cv2
import numpy as np


def get_nearest_interp_img(orig_img, dst_h, dst_w):

    height, width, channels = orig_img.shape  # 不需要转RGB因为不需要区分是哪个通道
    # height, width = orig_img.shape[ :2]   # 通道值可以不写 # 顺序是高，宽，通道，都是整型

    # print(orig_img)

    hm = dst_h / height  # High magnification factor
    wm = dst_w / width  # Wide magnification factor

    # 是不是一般uint8就够用了，那经过卷积等计算会超范围吗？
    nearest_img = np.zeros((dst_h, dst_w, 3), np.uint8)  # 必须要写通道数，否则通道数不对(默认是128)
    # print(nearest_img)

    # 不遍历通道实现：
    # for h in range(800):
    #     for w in range(800):
    #         y = int(h / hm + 0.5)  # 得到在原图中对应的最邻近(四舍五入)的点的坐标
    #         x = int(w / wm + 0.5)
    #         y0 = min(y, height)
    #
    #         nearest_img[h, w] = orig_img[y0, x0]  # 前面没有区分通道，这里也不用

    # 遍历通道实现
    channels = orig_img.shape[2]
    for h in range(dst_h):
        for w in range(dst_w):
            for c in range(channels):
                y = int(h / hm + 0.5)
                x = int(w / wm + 0.5)
                y0 = min(y, height-1)  # **这里应该和height-1比较，因为range(height)是0-511,y0的范围也应该是0-511
                x0 = min(x, width-1)
                nearest_img[h, w, c] = orig_img[y0, x0, c]

    return nearest_img


def get_bilinear_interp_img(img, out_size):
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = out_size[0], out_size[1]

    print(src_h, src_w)
    print(dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    # https://cloud.tencent.com/developer/article/2079083?areaId=106001
    hm = src_h / dst_h   # High magnification factor
    wm = src_w / dst_w   # Wide magnification factor
    # hm_ = (src_h-1) / (dst_h-1)
    # wm_ = (src_w-1) / (dst_w-1)

    print(hm, wm)

    channels = img.shape[2]

    for c in range(channels):
        for h in range(dst_h):
            for w in range(dst_w):
                src_y = (h + 0.5) * hm - 0.5
                src_x = (w + 0.5) * wm - 0.5

                # 整个图片对齐
                # src_y = h * hm_
                # src_x = w * wm_

                # 如果不做对齐
                # src_y = (h) * hm
                # src_x = (w) * wm

                # 取整，取到x0
                src_y0 = int(np.floor(src_y))
                src_x0 = int(np.floor(src_x))

                # if (src_x < 0): print(src_x, h, w, src_x0)

                # 取第二个x
                src_y1 = min(src_y0 + 1, src_h - 1)
                src_x1 = min(src_x0 + 1, src_w - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]  # y是高，在前
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                dst_img[h, w, c] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)  # TypeError: only size-1 arrays can be converted to Python scalars

    return dst_img


if __name__ == "__main__":

    original_img = cv2.imread("lenna.png")

    # 邻近插值【法一】：
    nearest_image = get_nearest_interp_img(original_img, 1024, 1024)

    # 邻近插值【法二】：
    # nearest_image = cv2.resize(original_img, (1024, 1024), cv2.INTER_NEAREST)

    # print(nearest_image)

    cv2.imshow("nearest_interp_img", nearest_image)

    # 双线性插值【法一】:
    bilinear_img = get_bilinear_interp_img(original_img, (800, 800))

    # 双线性插值【法二】：
    # bilinear_img = cv2.resize(original_img, (1024, 1024), cv2.INTER_LINEAR)

    cv2.imshow("bilinear_img", bilinear_img)
    cv2.waitKey(0)
