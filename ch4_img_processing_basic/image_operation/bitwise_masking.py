# -*- coding: utf-8 -*-
import numpy as np, cv2
import matplotlib.pylab as plt

# 이미지 읽기
img = cv2.imread('../img/my_girl1.jpg')

# 마스크 만들기
mask = np.zeros_like(img)
cv2.circle(mask, (450, 200), 180, (255, 255, 255), -1)
# cv2.circle(대상 이미지, (원점x, 원점y), 반지름, (색상), 채우기)

# 마스킹 (원)
masked = cv2.bitwise_and(img, mask)

# 결과 출력
cv2.imshow('original', img)
cv2.imshow('mask', mask)
cv2.imshow('masked', masked)
cv2.waitKey()
cv2.destroyAllWindows()