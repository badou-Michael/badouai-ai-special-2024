import numpy as np
from matplotlib import pyplot as plt


def normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

def z_score(x):
    x_mean=np.mean(x)
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]