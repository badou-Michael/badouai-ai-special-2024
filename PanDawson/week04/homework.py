'''GussianNoise
read image
set param
random_X random_Y
+GuassianNoise
show image
notice :the edge
'''
import numpy as np
import cv2
from numpy import shape 
import random
from PIL import Image
from skimage import util

img = cv2.imread('lenna.png',0)
means = 2
sigma = 4
percent = 0.8
random_X=random.randint(0,img.shape[0]-1) 
random_Y=random.randint(0,img.shape[1]-1)
img[random_X,random_Y]=img[random_X,random_Y]+random.gauss(means,sigma)

''' value maybe not in [0,255]'''
if  img[random_X, random_Y]< 0:
    img[random_X, random_Y]=0
elif img[random_X, random_Y]>255:
    img[random_X, random_Y]=255
img2 = cv2.cvtColor(cv2.imread('lenna.png'), cv2.COLOR_BGR2GRAY)
cv2.imshow('lenna_GaussianNoise',img)
cv2.imshow('lena_origin',img2)
cv2.waitKey(0)



'''
PepperandSalt Nosie
'''
img3 = cv2.imread('lenna.png',0)
percent = 0.8
random_X=random.randint(0,img3.shape[0]-1) 
random_Y=random.randint(0,img3.shape[1]-1)
if  random.random() <= 0.5:
    img3[random_X, random_Y]=0
elif random.random() > 0.5:
    img3[random_X, random_Y]=255
cv2.imshow('lenna_PepperandSalt',img3)
cv2.waitKey(0)



'''
other Noise
'''
img4 = cv2.imread("lenna.png")
img_poisson=util.random_noise(img4,mode='poisson')
img_gaussian=util.random_noise(img4,mode='gaussian')
img_PepperandSalt=util.random_noise(img4,mode='s&p')

cv2.imshow("source", img4)
cv2.imshow("img_poisson",img_poisson)
cv2.imshow("img_gaussian",img_gaussian)
cv2.imshow("img_PepperandSalt",img_PepperandSalt)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
PCA
read the martrix
get the mean
centralize
get the covariance matrix（D = 1/m * Z^T * Z）
get the eigenvector and eigenvalue (|A-aE| = 0)
get the eigen-matrix:W
new matrix = data-matrix * W
evaluation
'''
matirx = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
choose_trait = np.shape(matirx)[1] - 1 
column_means = np.mean(matirx, axis=0)
centralize_matrix = matirx - column_means
array_num = np.shape(centralize_matrix)[0]
cov_matrix = np.dot(centralize_matrix.T, centralize_matrix)/(array_num - 1)
eigenvalue,eigenvector = np.linalg.eig(cov_matrix)
index = np.argsort(-1*eigenvalue)
eigen_Matrix_T = []
k_eigenvalue=0
for i in range(choose_trait):
    for j in range(len(index)):
        if i == index[j]:
            eigen_Matrix_T.append(eigenvector[j])
            k_eigenvalue += eigenvalue[j]
eigen_Matrix = np.transpose(eigen_Matrix_T)
new_matrix =  np.dot(matirx, eigen_Matrix)
sum_eigenvalue = np.sum(eigenvalue)
evaluation = (k_eigenvalue/sum_eigenvalue)*100
evaluation

