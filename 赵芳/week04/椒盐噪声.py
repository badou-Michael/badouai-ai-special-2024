import cv2
import random
def  fun1(src,percetage):     
    NoiseImg=src    
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])    
    for i in range(NoiseNum): 
	  #每次取一个随机点 
    #random.randint生成随机整数
	    randX=random.randint(0,src.shape[0]-1)       
	    randY=random.randint(0,src.shape[1]-1) 
	    #生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0      
	    if random.random()<=0.5:           
	    	NoiseImg[randX,randY]=0       
	    else:            
	    	NoiseImg[randX,randY]=255    
    return NoiseImg
  
img=cv2.imread('lenna.png',0)
img1=fun1(img,0.8)
#在文件夹中写入命名为lenna_PepperandSalt.png的加噪后的图片
#cv2.imwrite('lenna_PepperandSalt.png',img1)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)
