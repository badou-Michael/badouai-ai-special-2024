# -*- coding: utf-8 -*-
# author: 王博然
import numpy as np
import cv2

def warpPerspectiveMatrix(src, dst):
    assert src.shape == dst.shape and src.shape[0] >= 4
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8)) # A * warpMatrix = B
    B = np.zeros((2 * nums, 1))
    for i in range(nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2 * i, :]     = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        A[2 * i + 1, :] = [0,0,0,A_i[0], A_i[1],1,-A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        
        B[2*i]    = B_i[0]
        B[2*i + 1] = B_i[1]

    A = np.matrix(A)
    warpMatrix = A.I * B

    warpMatrix = warpMatrix.A.T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis = 0)
    warpMatrix = warpMatrix.reshape(3,3)
    return warpMatrix

if __name__ == '__main__':
    dst_w = 377
    dst_h = 448
    # src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    src = np.float32([[151, 207], [285, 517], [601, 17], [731, 343]])    # 图片软件坐标和 np数组的x,y是相反的
    dst = np.float32([[0, 0], [0, dst_w], [dst_h, 0], [dst_h, dst_w]])
    src_img = cv2.imread('./photo1.jpg')
    # output_img = np.zeros((dst_h, dst_w, src_img.shape[2]), dtype=np.float32) 
    output_img = src_img.copy()[0:dst_h, 0:dst_w]

    # m = cv2.getPerspectiveTransform(src, dst)
    m = warpPerspectiveMatrix(src, dst)

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