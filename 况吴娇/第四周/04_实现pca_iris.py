#!/usr/bin/env python
# encoding=gbk

##import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import  load_iris


x ,y = load_iris(return_X_y=True) #�������ݣ�x��ʾ���ݼ��е��������ݣ�y��ʾ���ݱ�ǩ
pca=dp.PCA(n_components=2) #����pca�㷨�����ý�ά�����ɷ���ĿΪ2  #pca = sklearn.decomposition.PCA(n_components=2)
reduced_x=pca.fit_transform(x)  #��ά������� #fit_transform�������»���ȷʵ��ʾ�˷���ִ�е��Ⱥ�˳�򡣣�������reduced_x��

#1��ʼ�����б�
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
#2,������ά�����ݣ�
for i in range(len(reduced_x)): #���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
    #��ά���飬����ÿ�д���һ�����ݵ㣬ÿ�д���һ��������pca.fit_transform(x) �����ԭʼ���ݼ�ͨ�� PCA ��ά�������Ȼ��һ����ά���飬����ÿ�д���һ����ά������ݵ㣬ÿ�д���һ�����ɷ֡�
    #reduced_x�ĳ��ȣ���len(reduced_x)��ֵ������ԭʼ���ݼ��е����ݵ�������

    # �����ݼ��У������ͱ�ǩͨ����������ʽ��֯��
    # ��������һ����ά���飬����ÿһ�д���һ��������ÿһ�д���һ��������
    # ��ǩ���飺һ��һά���飬���е�ÿ��Ԫ�ض�Ӧ����������ÿһ�������ı�ǩ����ǩ����:['Setosa', 'Sonata', 'Versicolor']   /[0, 1, 2] 0 ���� "Setosa"��1 ���� "Sonata"��2 ���� "Versicolor"
#1��ʼ�����б���Щ�б����ڴ洢��ͬ�����β���Ľ�ά�����ꡣ
    # reduced_x[i]������ʾ�� i ����ά������ݵ㣨һ���������������������б���
    # reduced_x[i][0]����ʾ��i�����ݵ�ĵ�һ��������
    # reduced_x[i][1]����ʾ��i�����ݵ�ĵڶ���������
    if y[i] == 0:  #���ݱ�ǩ��y[i]�������жϣ���Ϊ�������ÿ�����ݵ�ġ���ȷ�𰸡���������β�����ݼ��������У���ǩ��������ÿ�����������ĸ�������β����
        red_x.append(reduced_x[i][0])##�ڻ�ͼʱʹ�� append ����������Ϊ����ͨ����Ҫ�����ݵ�������ռ����б��У��Ա��������һ���Եؽ����ǻ��Ƶ�ͼ���ϡ�
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:####��������ͨ��PCA��ά��2�����ɷ֣����ݼ��������3�������β��
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()


 # sklearn.decomposition��scikit-learn�����sklearn�����е�һ��ģ�飬���ṩ��һϵ���������ݽ�ά��������ȡ�ķ�����
# # PCA (Principal Component Analysis): ���ɷַ���������ͨ�����Ա任������ͶӰ���µ�����ϵ�У��Բ�׽���������ķ��

# ���� return_X_y �� Scikit-learn �����ݼ��غ�����һ������������ָ�������������ݵķ�ʽ��������Ϊ True ʱ���������������ֿ������飺X���������ݣ��� y��Ŀ�����ݻ��ǩ����
# �������еĴ�д��ĸ X ��Сд y ��Ϊ����������������Ĳ�ͬ���֡�X ͨ��������ʾ�������ݼ����� y ��ʾĿ�����ݼ����ǩ���顣