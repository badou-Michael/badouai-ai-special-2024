# -*- coding: utf-8 -*-
# author: 王博然
import numpy as np
import cv2

def warpPerspectiveMatrix(src, dst):
    pass   # todo: 自行实现一个

if __name__ == '__main__':
    dst_w = 377
    dst_h = 448
    # src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    src = np.float32([[151, 207], [285, 517], [601, 17], [731, 343]])    # 图片软件坐标和 np数组的x,y是相反的
    dst = np.float32([[0, 0], [0, dst_w], [dst_h, 0], [dst_h, dst_w]])
    src_img = cv2.imread('./photo1.jpg')
    # output_img = np.zeros((dst_h, dst_w, src_img.shape[2]), dtype=np.float32) 
    output_img = src_img.copy()[0:dst_h, 0:dst_w]

    m = cv2.getPerspectiveTransform(src, dst)

    for x in range(src_img.shape[0]):
        for y in range(src_img.shape[1]):
            
            origin_p = np.array([x,y,1], dtype=np.float32)
            trans_p = m @ origin_p

            z = trans_p[2]
            if z != 0:
                x_final = int(trans_p[0]/z)
                y_final = int(trans_p[1]/z)

            if 0 <= x_final < dst_h and 0 <= y_final < dst_w:
                output_img[x_final, y_final] = src_img[x, y]

    cv2.imshow("src", src_img)
    cv2.imshow("result", output_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()