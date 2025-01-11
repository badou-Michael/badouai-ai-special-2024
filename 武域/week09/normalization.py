import numpy as np
import matplotlib.pyplot as plt

def normalization(data):
    return (data - min(data)) / (max(data) - min(data))

def z_score(data):
    return (data - data.mean()) / data.std()

l=np.array([-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30])
unique_values, counts = np.unique(l, return_counts=True)
n = normalization(l)
z = z_score(l)

# Print normalized data
print("Normalized Data (Min-Max):", n)
print("Normalized Data (Z-Score):", z)