# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
f1 = x*5
f2 = x**2
f3 = x**2 + x*2

plt.plot(x, 'r--')# 빨간색 이음선
plt.plot(f1, 'g.')#녹색 점선
plt.plot(f2, 'bv')#파란색 역삼각형
plt.plot(f3, 'ks')#검은색 사각형
plt.show()