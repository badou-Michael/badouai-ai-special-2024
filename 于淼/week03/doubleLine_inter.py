import numpy as np
import cv2


def doubleLine_insert(img, out_dim):  # 输入图片和图像大小
    sh, sw, channels = img.shape
    aim_h, aim_w = out_dim[1], out_dim[0]
    print("原图高，原图宽=", sh, sw)
    print("目标图高，宽 =", aim_h, aim_w)
    if sh == aim_h and sw == aim_w:  # 原始图像和目标图像尺寸一样，直接copy
        return img.copy()
    aim_img = np.zeros((aim_h, aim_w, 3), dtype=np.uint8)  # 创建图像
    scale_y, scale_x = float(sh) / aim_h, float(sw) / aim_w  # 缩放比例

    for i in range(channels):
        for aim_y in range(aim_h):
            for aim_x in range(aim_w):
                sh_x = (aim_x + 0.5) * scale_x - 0.5
                sh_y = (aim_y + 0.5) * scale_y - 0.5

                #sh_x0 = int(np.floor(sh_x))        # np.floor()————向下取整
                sh_x0 = int(sh_x)
                sh_x1 = min(sh_x0 + 1, sw - 1)
                #sh_y0 = int(np.floor(sh_y))
                sh_y0 = int(sh_y)
                sh_y1 = min(sh_y0 + 1, sh - 1)

                temp0 = (sh_x1 - sh_x) * img[sh_y0, sh_x0, i] + (sh_x - sh_x0) * img[sh_y0, sh_x1, i]
                temp1 = (sh_x1 - sh_x) * img[sh_y1, sh_x0, i] + (sh_x - sh_x0) * img[sh_y1, sh_x1, i]
                aim_img[aim_y, aim_x, i] = int((sh_y1 - sh_y) * temp0 + (sh_y - sh_y0) * temp1)
    return aim_img


img = cv2.imread("F:\DeepLearning\Code_test\lenna.png")
aim_img = doubleLine_insert(img, (800, 800))
print("目标图尺寸：", aim_img.shape)
cv2.imshow("doubleImg",aim_img)
cv2.imshow("sImg",img)
cv2.waitKey()

