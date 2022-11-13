# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('../img/gray_gradient.jpg', cv2.IMREAD_GRAYSCALE)

'''
ret, out = cv2.threshold(img, threshold, value, type_flag)
 - img: NumPy 배열, 변환할 이미지
 - threshold: 경계 값
 - value: 경계값 기준에 만족하는 픽셀에 적용할 값
 - type_flag: 스레시홀드 적용 방법 지정
'''
# 픽셀 값이 경계 값을 넘으면 value를 지정하고, 넘지 못하면 0을 지정
_, t_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# cv2.THRESH_BINARY의 반대
_, t_bininv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# 픽셀 값이 경계 값을 넘으면 value를 지정하고, 넘지 못하면 원래의 값 유지
_, t_truc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# 픽셀 값이 경계 값을 넘으면 원래 값을 유지, 넘지 못하면 0을 지정
_, t_2zr = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# cv2.THRESH_TOZERO의 반대
_, t_2zrinv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

imgs = {'origin': img, 'BINARY': t_bin, 'BINARY_INV':t_bininv, \
    'TRUNC': t_truc, 'TOZERO': t_2zr, 'TOZERO_INV': t_2zrinv}

for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(2, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()
