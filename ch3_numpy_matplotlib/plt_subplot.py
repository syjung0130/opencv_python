# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)

plt.subplot(2, 2, 1) ## 2행 2열 중 첫번째
plt.plot(x, x**2)

plt.subplot(2, 2, 2) ## 2행 2열 중 두번째
plt.plot(x, x*5)

plt.subplot(223) ## 2행 2열 중 세번째
plt.plot(x, np.sin(x))

plt.subplot(224) ## 2행 2열 중 네번째
plt.plot(x, np.cos(x))

plt.show()