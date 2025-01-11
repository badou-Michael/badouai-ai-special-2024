import numpy as np
import matplotlib.pyplot as plt

def Nomalization(x):
    '''x_=(x−x_min)/(x_max−x_min)'''
    '''0-1'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]

def z_score(x):
    mean = np.mean(x)
    s2 = sum([(i - np.mean(i) ** 2) for i in x])/len(x) # 方差
    return [(i - mean) / s2 for i in x]

