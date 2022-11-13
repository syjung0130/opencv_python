# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('..\img\my_girl1.jpg')
# 0~255까지의 숫자를 더하면 8비트 범위를 벗어나기 때문에 uint16으로 변환한다.
img2 = img.astype(np.uint16)
b, g, r = cv2.split(img2)# 채널 별로 분리
gray1 = ((b+g+r)/3).astype(np.uint8) #평균값 연산 후 dtype변경

gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('original', img)
cv2.imshow('gray1', gray1)
cv2.imshow('gray2', gray2)

cv2.waitKey(0)
cv2.destroyAllWindows()