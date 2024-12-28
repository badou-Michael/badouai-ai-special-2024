import numpy as np
import matplotlib.pyplot as plt


def min_max(arr):
    arr = np.array(arr)
    min_num = arr.min()
    max_num = arr.max()
    return (arr - min_num) / max_num


def z_score(arr):
    arr = np.array(arr)
    mean_num = arr.mean()
    s = np.sum((arr - mean_num) ** 2) / arr.shape[0]
    return (arr - mean_num) / s
