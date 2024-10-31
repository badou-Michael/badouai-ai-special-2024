import random # 导入模块
import cv2

def JiaoYan(scr,percentage):
    JiaoYanNoise=scr
    JiaoYanNum=int(scr.shape[0]*scr.shape[1]*percentage)# 要加噪的像素数目
    for i in range(JiaoYanNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        randX=random.randint(0,scr.shape[0]-1)# 随机选取的坐标点 -行
        randY=random.randint(0,scr.shape[1]-1)# 随机选取的坐标点 -列
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() <= 0.5: # Random.random [0,1]之间的随机数
           JiaoYanNoise[randX,randY]=0
        elif random.random()>0.5:
            JiaoYanNoise[randX, randY] = 1
    return JiaoYanNoise

image=cv2.imread('lenna.png',0)
JiaoYanNoise=JiaoYan(image,0.8)
cv2.imshow("show JiaoYanNoise:",JiaoYanNoise)
cv2.waitKey(0)
