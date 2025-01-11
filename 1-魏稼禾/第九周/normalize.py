import numpy as np
import matplotlib.pyplot as plt

def normal1(input):
    return [(i-np.min(input))/(np.max(input)-np.min(input)) for i in input]

def normal2(input):
    return [(i-np.mean(input)/(np.max(input)-np.min(input))) for i in input]

def z_score(input):
    sigma = np.sum([(i-np.mean(input))**2 for i in input])/len(input)
    return [(i-np.mean(input))/sigma for i in input]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
sc=[]
for i in l:
    c = l.count(i)
    sc.append(c)
# l1 = normal1(l)
l1 = z_score(l)
plt.plot(l,sc)
plt.plot(l1,sc)
plt.show()