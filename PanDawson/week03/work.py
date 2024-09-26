'''
image1 M*M  central: ((M-1)/2,(M-1)/2)
image2 N*N  central: ((N-1)/2,(N-1)/2)
(N-1)/2 + Z = ((M-1)/2 + Z)*(N/M)
(1-N/M) * Z = (M-N)/(2*M)
Z=1/2
'''

'''
nearest 最邻近插值法
read image
newx,newy,image_ch
scale_x,scale_y
for i in range newx
    for j in range newy :
src_x=int(newx/scale_x+0.5)
src_y
new_image[i,j] = image[src_x,src_y]
    end for
end for
show new_image
'''

import cv2
import numpy as np

image=cv2.imread("lenna.png")
height,width,channels =image.shape
new_height = 800
new_width = 800
new_image = np.zeros((new_width,new_height,channels),np.uint8)
scale_x = new_width / width
scale_y = new_height / height
for i in range(new_width):
        for j in range(new_height):
            x=int(i/scale_x + 0.5) 
            y=int(j/scale_y + 0.5)
            new_image[i,j]=image[x,y]
          
'''
bilinear 双线插值
read image
newx newy image_ch
scale_x scale_y
for k in image_ch
    for i in newx
        for j in newy
src_x=(i+0.5)/scale_x-0.5
src_y
src_x0 =int src_x
src_y0=int src_y
src_x1=min src_x0+1,image_x-1
src_y1=min
new_image[i,j,k] = format_calculate
end all for
show image
'''

def bilinear_calculation(source_x,source_y,image,source_height,source_width,current_channel):
    source_x1 = int(source_x)     
    source_x2 = min(source_x1 + 1 ,source_width - 1)
    source_y1 = int(source_y)
    source_y2 = min(source_y1 + 1, source_height - 1)
    
    r1 = (source_x2 - source_x) * image[source_x1,source_y1,current_channel] + (source_x - source_x1) * image[source_x2,source_y1,current_channel]
    r2 = (source_x2 - source_x) * image[source_x1,source_y2,current_channel] + (source_x - source_x1) * image[source_x2,source_y2,current_channel]
    new_channel = int((source_y2 - source_y) * r1 + (source_y - source_y1) * r2)
    return new_channel

bilinear_image = np.zeros((new_width,new_height,channels),dtype=np.uint8)
for current_channel in range(channels):
    for current_y in range(new_height):
        for current_x in range(new_width):
            source_x = (current_x + 0.5) / scale_x-0.5
            source_y = (current_y + 0.5) / scale_y-0.5
            bilinear_image[current_x,current_y,current_channel] = bilinear_calculation(source_x,source_y,image,height,width,current_channel)
cv2.imshow("the lena image",image)          
cv2.imshow("after nearest interp image",new_image)
cv2.imshow('after bilinear interp image',bilinear_image)
cv2.waitKey(0)

'''
灰度均衡化
'''
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.equalizeHist(gray)
cal_hist = cv2.calcHist([hist],[0],None,[256],[0,256])
cv2.imshow("Histogram Equalization", np.hstack([gray, hist]))
cv2.waitKey(0)
