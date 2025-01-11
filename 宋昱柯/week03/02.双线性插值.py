import cv2
import numpy as np


def bilinear_interp(img, new_height, new_width):
    """双线性插值"""
    height, width, channels = img.shape

    if (new_height, new_width) == (height, width):
        return img.copy()

    new_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    scale_h, scale_w = height / new_height, width / new_width

    for c in range(channels):
        for h in range(new_height):
            for w in range(new_width):
                # 几何中心点对齐
                src_w = scale_w * (w + 0.5) - 0.5
                src_h = scale_h * (h + 0.5) - 0.5

                src_w0 = int(src_w)
                src_w1 = min(src_w0 + 1, width - 1)
                src_h0 = int(src_h)
                src_h1 = min(src_h0 + 1, height - 1)

                interp_w1 = (src_w1 - src_w) * img[src_h0, src_w0, c] + (
                    src_w - src_w0
                ) * img[src_h0, src_w1, c]
                interp_w2 = (src_w1 - src_w) * img[src_h1, src_w0, c] + (
                    src_w - src_w0
                ) * img[src_h1, src_w1, c]
                new_img[h, w, c] = (src_h1 - src_h) * interp_w1 + (
                    src_h - src_h0
                ) * interp_w2

    return new_img


if __name__ == "__main__":
    img = cv2.imread("practice/cv/week03/lenna.png")
    new_img = bilinear_interp(img, 800, 800)
    # new_img=cv2.resize(img,(800,800),interpolation=cv2.INTER_LINEAR)
    cv2.imshow("bilinear interp", new_img)
    cv2.waitKey(0)
