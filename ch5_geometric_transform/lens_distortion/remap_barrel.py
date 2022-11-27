# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 왜곡 계수 설정
k1, k2, k3 = 0.5, 0.2, 0.0 # 배럴 왜곡
#k1, k2, k3 = -0.3, 0, 0 # 배럴 왜곡

img = cv2.imread('../img/my_girl1.jpg')
rows, cols = img.shape[:2]

# 매핑 배열 생성
mapy, mapx = np.indices((rows, cols), dtype=np.float32)

# 중앙점 좌표로 -1~1 정규화 및 극좌표 변환
mapx = 2*mapx/(cols-1) -1
mapy = 2*mapy/(rows-1) -1
r, theta = cv2.cartToPolar(mapx, mapy)

# 방사 왜곡 변형 연산
ru = r*(1+k1*(r**2) + k2*(r**4) + k3*(r**6))

# 직교좌표 및 좌상단 기준으로 복원
mapx, mapy = cv2.polarToCart(ru, theta)
mapx = ((mapx + 1)*cols -1) / 2
mapy = ((mapy + 1)*rows -1) / 2

# 리매핑
distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow('original', img)
cv2.imshow('distorted', distorted)
cv2.waitKey()
cv2.destroyAllWindows()