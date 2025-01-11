from skimage import util

img=cv2.imread("lenna.png")
noise_img=util.random_noise(img,mode="salt")
noise_ps_img=util.random_noise(img,mode='poisson')
noise_gs_img=util.random_noise(img,mode='gaussian')
cv2.imshow("noise_img",noise_img)
cv2.imshow("noise_ps_img",noise_ps_img)
cv2.imshow("noise_gs_img",noise_gs_img)
cv2.imshow("img",img)
cv2.waitKey(0)
