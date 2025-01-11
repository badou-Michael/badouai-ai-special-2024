def gaosi(img,means,sigma,percetage):
	image = img
	imgnum = (percetage*img.shape[0]*img.shape[1])
	for i in range(imgnum):
		randx = random.raandint(0,img.shape[1]-1)
		randy = random.raandint(0,img.shape[0]-1)
		image[randx,randy] = image[randx,randy]+random.gauss(means,sigma)
		if image[randx,randy]>255:
			image[randx,randy]=255
		if image[randx,randy]<0:
			image[randx,randy]=0
	return image
img = cv2.imread('leanna.png',0 )
img1 = gaosi(img,2,4,0.8)
img = cv2.imread('leanna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('1',img2)
cv2.imshow('gaosi',img1)
cv2,waitKey(0)
def jiaoyan(img,percetage):
	image = img
	imgnum = (percetage*img.shape[0]*img.shape[1])
	for i in range(imgnum):
		randx = random.raandint(0,img.shape[1]-1)
		randy = random.raandint(0,img.shape[0]-1)
		if random.random()<=0.5:
			image[randx,randy]=255
		else:
			image[randx,randy]=0
	return image
img = cv2.imread('leanna.png',0 )
img1 = jiaoyan(img,0.8)
img = cv2.imread('leanna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('1',img2)
cv2.imshow('jiaoyan',img1)
cv2,waitKey(0)
x,y = load_iris(return_X_y=True)
pca = dp.PCA(n_components=2)
reduced_x=pca.flt_transform(x)
red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): 
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
