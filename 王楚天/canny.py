import cv2
import numpy as np
def canny(img,sigma,high_threshold,low_threshold):
    h,w,_=img.shape
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel_size=(3,3)
    kernel=cv2.getGaussianKernel(kernel_size[0],sigma)
    gauss_kernel=np.outer(kernel,kernel.transpose())
    gauss_kernel/=np.sum(gauss_kernel)
    gauss_img=np.zeros((h-2,w-2),dtype=float)
    for i in range(0,w-2):
        for j in range(0,h-2):
            cur_region=img[i:i+3,j:j+3]
            gauss_img[i][j]=np.sum(gauss_kernel*cur_region)
    sobel_x = cv2.Sobel(gauss_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gauss_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    yu_low,yu_high=127,255
    low_empty_img=np.zeros_like(gradient_magnitude)
    high_empty_img=np.zeros_like(gradient_magnitude)
    high_empty_img[gradient_magnitude>=high_threshold]=yu_high
    low_empty_img[(gradient_magnitude>=low_threshold) & (gradient_magnitude<high_threshold)]=yu_low
    combined=np.zeros_like(gradient_magnitude)
    combined[high_empty_img>0]=yu_high
    combined[low_empty_img>0]=yu_low
    n,m=combined.shape
    for i in range(1,n-1):
        for j in range(1,m-1):
            if combined[i,j]==yu_low:
                if combined[i-1,j]==yu_high or combined[i+1,j]==yu_high or combined[i,j+1]==yu_high or combined[i,j-1]==yu_high :
                    combined[i,j]=yu_high
                else:
                    combined[i,j]=0
    return combined

img=cv2.imread("lenna.png")
sigma=1.5
output_img=canny(img,sigma,100,30)
cv2.imshow('Gaussian Filtered Image', output_img.astype(np.uint8))
cv2.waitKey(0)



