# -*- coding: utf-8 -*-
import cv2

'''
회전 변환행렬 구하기
'''

img = cv2.imread('../img/fish.jpg')
rows, cols = img.shape[0:2]

# 회전을 위한 변환행렬 구하기
# 회전 축: 중앙, 각도: 45, 배율: 0.5
m45 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)
# 회전 축: 중앙, 각도: 90, 배율: 1.5
m90 = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1.5)

# 변환행렬 적용
img45 = cv2.warpAffine(img, m45, (cols, rows))
img90 = cv2.warpAffine(img, m90, (cols, rows))

# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('45', img45)
cv2.imshow('90', img90)
cv2.waitKey(0)
cv2.destroyAllWindows()