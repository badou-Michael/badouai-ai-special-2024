import numpy as np
import cv2

def nni(img, target_H, target_W):
    H_ori = img.shape[0]
    W_ori = img.shape[1]
    img_new = np.zeros((target_H, target_H, img.shape[2]), np.uint8)
    hs = target_H/H_ori
    ws = target_W/W_ori
    for h in range(target_H):
        for w in range(target_W):
            h_ori = min(int(h/hs+0.5), H_ori-1)
            w_ori = min(int(w/ws+0.5), W_ori-1)
            img_new[h,w] = img[h_ori,w_ori]
    return img_new

def bi(img, target_H, target_W):
    ori_H = img.shape[0]
    ori_W = img.shape[1]
    if target_H == ori_H and target_W == ori_W:
        return img.copy()
    img_new = np.zeros((target_H, target_W, 3), np.uint8)
    hs = target_H / ori_H
    ws = target_W / ori_W
    for i in range(img.shape[2]):
        for h in range(target_H):
            for w in range(target_W):
                ori_h = (h+0.5)/hs-0.5
                ori_w = (w+0.5)/ws-0.5
                ori_h0 = int(np.floor(ori_h))   #感觉floor不是很有必要
                ori_w0 = int(np.floor(ori_w))
                ori_h1 = min(ori_h0+1, ori_H-1)
                ori_w1 = min(ori_w0+1, ori_W-1)
                p0 = (ori_w1-ori_w)*img[ori_h0, ori_w0, i]+(ori_w-ori_w0)*img[ori_h0,ori_w1, i]   #除数可以省略
                p1 = (ori_w1-ori_w)*img[ori_h1, ori_w0, i]+(ori_w-ori_w0)*img[ori_h1,ori_w1, i]
                q = int((ori_h1-ori_h)*p0 + (ori_h - ori_h0)*p1)
                img_new[h,w,i] = q
    return img_new

if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    img_nni = nni(img, 1200, 1200)
    img_bi = bi(img, 1200, 1200)
    # img_bi = cv2.resize(img, (1200,1200), interpolation=cv2.INTER_LINEAR) #建议使用
    cv2.imshow("ori", img)
    cv2.imshow("nearest interp", img_nni)
    cv2.imshow("bilinear interp", img_bi)
    cv2.waitKey()
    cv2.destroyAllWindows()