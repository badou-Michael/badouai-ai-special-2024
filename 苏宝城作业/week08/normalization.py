import numpy as np
import matplotlib.pyplot as plt

def Normalization1(x):   #输入数字列表，将数字列表里面的数据归一化到区间（0，1）内
  res = []
  for i in x:
    ans = (float(i) - min(x)) / float(max(x) - min(x))
    res.append(ans)
  return res

def Normalization2(x):   #输入数字列表，将数字列表里面的数据归一化到区间（-1，1）内
  res = []
  for i in x:
    ans = (float(i) - np.mean(x)) / float(max(x) - min(x))
    res.append(ans)
  return res

def z_score(x):   #输入数字列表，将数据标准化
  x_mean = np.mean(x)
  res = []
  for i in x:
    s = sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    ans = (i - x_mean) / s
    res.append(ans)
  return res


  
  
