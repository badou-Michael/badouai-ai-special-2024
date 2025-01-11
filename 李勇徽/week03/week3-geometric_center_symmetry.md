-  原图像

  - 尺寸:  $M\times M$,

  - 原图像坐标: $(x_m, y_m)$

  - 原图像几何中心: 
    $$
    \left(x_{\frac{M-1}{2}},y_{\frac{M-1}{2}}
    \right)
    \tag {1}
    $$
    

- 目标图像:

  - 尺寸: $N \times N$

  - 目标图像坐标: $(x_n, y_n)$

  - 目标图像几何中心:  
    $$
    \left(x_{\frac{N-1}{2}},y_{\frac{N-1}{2}}
    \right)
    \tag {2}
    $$
    

- 缩放比例

  - 原图像 -> 目标图像: $\frac{N}{M}$

  - 坐标对应: 
    $$
    m \times \frac{N}{M}=n
    \tag{3}
    $$
    

- 几何中心重合

  设置图像平移`z`个单位，并使两个图像几何中心重合，那么由$(1)(2)(3)$可得：
  $$
  (\frac{M-1}{2} + z) \times \frac{N}{M} = \frac{N-1}{2}+z
  \tag{4}
  $$
  由此可得，$z=\frac{1}{2}$



