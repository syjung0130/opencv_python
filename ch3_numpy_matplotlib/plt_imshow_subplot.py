# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2

img1 = cv2.imread('.\img\my_girl1.jpg')
img2 = cv2.imread('.\img\my_girl2.jpg')
img3 = cv2.imread('.\img\my_girl3.jpg')

plt.subplot(1, 3, 1) # 1행 3열 중에 첫번째
plt.imshow(img1[:,:,(2,1,0)])
plt.xticks([]); plt.yticks([])

plt.subplot(1, 3, 2) # 1행 3열 중에 두번째
plt.imshow(img2[:,:,::-1])
plt.xticks([]); plt.yticks([])

plt.subplot(1, 3, 3) # 1행 3열 중에 세번째
plt.imshow(img3[:,:,::-1])
plt.xticks([]); plt.yticks([])

plt.show()