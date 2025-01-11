import cv2
import numpy as np

def bilinear_interpolation(img, method=1):
    if method == 1:
        s_h, s_w, channels = img.shape
        d_h, d_w = 800, 800
        scale_x, scale_y = float(s_h)/d_h, float(s_w)/d_w
        img_zoom = np.zeros((800, 800, 3), np.uint8)

        for c in range(channels):
            for i in range(d_h):
                for j in range(d_w):
                    # find the corresponding (x,y) in source image
                    # use geometric center symmetry
                    s_x = (i + 0.5) * scale_x - 0.5
                    s_y = (j + 0.5) * scale_y - 0.5

                    # find the corresponding points to compute the interpolation
                    s_x0 = int(np.floor(s_x))
                    s_y0 = int(np.floor(s_x))
                    s_x1 = min(s_x0 + 1, s_h - 1)
                    s_y1 = min(s_y0 + 1, s_w - 1)

                    # calculate the interpolation
                    f_R1 = (s_x - s_x0) * img[s_x1,s_y0,c] + (s_x1 - s_x) * img[s_x0,s_y0,c]
                    f_R2 = (s_x - s_x0) * img[s_x1,s_y1,c] + (s_x1 - s_x) * img[s_x0,s_y1,c]

                    img_zoom[i, j, c] = int((s_y1 - s_y) * f_R1 + (s_y - s_y0) * f_R2)
    elif method == 2:
        img_zoom = cv2.resize(img, (800, 800), cv2.INTER_LINEAR)
    cv2.imshow('bilinear interpolation', img_zoom)
    cv2.waitKey(0)

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    bilinear_interpolation(img, 2)
