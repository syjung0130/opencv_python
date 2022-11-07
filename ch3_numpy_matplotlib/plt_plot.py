# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

a = np.array([2, 6, 7, 3, 12, 8, 4, 5]) # 배열 생성
plt.plot(a) # plot 생성
plt.show() # plot 그리기

## 1차원 배열을 인자로 전달하면
## 배열의 인덱스를 x 좌표로,
## 배열의 값을 y좌표로 써서 그래프를 그린다.