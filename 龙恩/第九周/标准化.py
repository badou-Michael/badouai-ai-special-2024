import numpy as np
import matplotlib.pyplot as plt
import random

def std(x):
    mean=np.mean(x)
    sigma2=sum([(i-mean)*(i-mean) for i in x])/len(x)
    return [(i-mean)/sigma2 for i in x]

#l=[random.randint(-2, 11) for i in range(50)]
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9,
   9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
   11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l.sort()
x=[]

for i in l:
    c=l.count(i)
    x.append(c)

s=std(l)

plt.plot(l,x,label='Original')
plt.plot(s,x,label='zero mean')
plt.legend()
plt.show()


    
