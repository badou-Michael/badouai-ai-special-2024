import matplotlib.pyplot as plt
import numpy as np

#三种标准化的两种方式
def Normalization1(x):
    '''归一化（0~1）：x_new=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]
def Normalization2(x):
    '''归一化（-1~1）：x_new=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]
def z_score(x):
    '''z_score标准化：x_new=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2=sum((i - x_mean)*(i - x_mean) for i in x)/len(x)
    return [(i - x_mean)/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
print('l:',l[:5])

l1=[]
l1=[(i+2) for i in l]
print('l1:',l1[:5])

cs=[]
for i in l:
    co = l.count(i)
    cs.append(co)
print('cs:',cs[:5])

n1=Normalization1(l)
n2=Normalization2(l)
n3=z_score(l)
print('n1:',n1[:5])
print('n2:',n2[:5])
print('n3:',n3[:])

# plt.plot(l,cs,label='l')
# plt.plot(l1,cs,label='l-1')
plt.plot(n1,cs,label='n1')
plt.plot(n2,cs,label='n2')
plt.plot(n3,cs,label='n3')
plt.legend()
plt.show()
