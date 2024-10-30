import numpy as np
import matplotlib.pyplot as plt
import math
 #read the image and to gray
img = plt.imread("lenna.png")
img *= 255
img = img.mean(axis=-1)
plt.figure(1)
plt.imshow(img.astype(np.uint8), cmap='gray') 
plt.axis('off')

#Gaussian
#param
sigma = 0.8
dim = 5
Gaussian_filter = np.zeros([dim,dim])
distanceToTheCenter_seq = [i-dim//2 for i in range(dim)]
# 计算高斯核  中间最大，越向边缘越小
# get the Guassian_filter 
n1 = 1/(2*math.pi*sigma**2) 
n2 = -1/(2*sigma**2)
for i in range(dim):  # 0-4
    for j in range(dim):  # 0-4  temp = [-2,-1,0,1,2]
        Gaussian_filter[i, j] = n1*math.exp(n2*(distanceToTheCenter_seq[i]**2+distanceToTheCenter_seq[j]**2))
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

# get the new image after Gaussian_filter
# pad the old image for Gaussian_filter to calculate
new_border_width = dim//2 
img_pad = np.pad(img, ((new_border_width, new_border_width), (new_border_width, new_border_width)), 'constant')

# use a new image to store the value
img_new = np.zeros(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)

plt.figure(2)
plt.imshow(img_new.astype(np.uint8), cmap='gray')  
plt.axis('off')

# get the gradient matrix by sobel
#set the sobel
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

#have a new array to store the gradient result
img_tidu_x = np.zeros(img_new.shape)  
img_tidu_y = np.zeros(img_new.shape)
img_tidu = np.zeros(img_new.shape)

# soble is 3*3 so need to pad the img_new for better calculation
img_pad_for_sobel = np.pad(img_new, ((1, 1), (1, 1)), 'constant') 

#calculate by sobel
for i in range(img_new.shape[0]):
    for j in range(img_new.shape[1]):
        img_tidu_x[i, j] = np.sum(img_pad_for_sobel[i:i+3, j:j+3]*sobel_kernel_x) 
        img_tidu_y[i, j] = np.sum(img_pad_for_sobel[i:i+3, j:j+3]*sobel_kernel_y) 
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)

plt.figure(3)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')
#NMS Non-Maximun Suppresion
#param
#judge the tmp1 tmp2 with C in the ppt
#to get tmp1 tmp2 we need angle and the two-line insertion
img_tidu_x[img_tidu_x == 0] = 0.00000001  # to calculate the angle
angle = img_tidu_y/img_tidu_x
img_NMS = np.zeros(img_tidu.shape)

for i in range(1, img_NMS.shape[0]-1):
    for j in range(1, img_NMS.shape[1]-1):
        temp = img_tidu[i-1:i+2, j-1:j+2] 
        if angle[i, j] <= -1:  
            tmp1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            tmp2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if (img_tidu[i, j] > tmp1 and img_tidu[i, j] > tmp2):
                img_NMS[i, j] = img_tidu[i, j]
        elif angle[i, j] >= 1:
            tmp1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            tmp2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if (img_tidu[i, j] > tmp1 and img_tidu[i, j] > tmp2):
                img_NMS[i, j] = img_tidu[i, j]
        elif angle[i, j] > 0:
            tmp1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            tmp2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if (img_tidu[i, j] > tmp1 and img_tidu[i, j] > tmp2):
                img_NMS[i, j] = img_tidu[i, j]
        elif angle[i, j] < 0:
            tmp1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            tmp2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if (img_tidu[i, j] > tmp1 and img_tidu[i, j] > tmp2):
                img_NMS[i, j] = img_tidu[i, j]
plt.figure(4)
plt.imshow(img_NMS.astype(np.uint8), cmap='gray')
plt.axis('off')
#two-thresholds to disguish the strong edge and the weak edge
#set the thresholds
lower_boundary = img_tidu.mean() * 0.75
high_boundary = lower_boundary * 3 
# a stack to instore all strong edge 
strongEdge_Stack = []
# find all strong edge and non-edge
# if the weak edge near the strong one , change the weak one to strong one 
# if not near,set to 0 
for i in range(1, img_NMS.shape[0]-1): 
    for j in range(1, img_NMS.shape[1]-1):
        if img_NMS[i, j] >= high_boundary: 
            img_NMS[i, j] = 255
            strongEdge_Stack.append([i, j])
        elif img_NMS[i, j] <= lower_boundary: 
            img_NMS[i, j] = 0


#judge the 8 points near the strong edge point 
#if the point is a weak one , set it to a strong one 
while not len(strongEdge_Stack) == 0:
    c_x, c_y = strongEdge_Stack.pop()
    a = img_NMS[c_x-1:c_x+2, c_y-1:c_y+2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_NMS[c_x-1, c_y-1] = 255 
        strongEdge_Stack.append([c_x-1, c_y-1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_NMS[c_x - 1, c_y] = 255
        strongEdge_Stack.append([c_x - 1, c_y])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_NMS[c_x - 1, c_y + 1] = 255
        strongEdge_Stack.append([c_x - 1, c_y + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_NMS[c_x, c_y - 1] = 255
        strongEdge_Stack.append([c_x, c_y - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_NMS[c_x, c_y + 1] = 255
        strongEdge_Stack.append([c_x, c_y + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_NMS[c_x + 1, c_y - 1] = 255
        strongEdge_Stack.append([c_x + 1, c_y - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_NMS[c_x + 1, c_y] = 255
        strongEdge_Stack.append([c_x + 1, c_y])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_NMS[c_x + 1, c_y + 1] = 255
        strongEdge_Stack.append([c_x + 1, c_y + 1])

for i in range(img_NMS.shape[0]): 
    for j in range(img_NMS.shape[1]):
        if img_NMS[i, j] != 0 and img_NMS[i, j] != 255:
            img_NMS[i, j] = 0

plt.figure(5)
plt.imshow(img_NMS.astype(np.uint8), cmap='gray')
plt.axis('off')  
plt.show()
