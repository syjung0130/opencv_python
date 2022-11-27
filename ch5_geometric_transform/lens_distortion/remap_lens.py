# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('../img/taekwonv1.jpg')
rows, cols = img.shape[:2]

# 설정 값 세팅
exp = 0.5#2#0.5 # 블록, 오목 지수(오목: 0.1~1, 볼록: 1.1~)
scale = 1 # 변환 영역 크기(0~1)

# 매핑 배열 생성
mapy, mapx = np.indices((rows, cols), dtype=np.float32)

# 좌상단 기준좌표에서 -1 ~ 1 로 정규화된 중심점 기준 좌표로 변경
mapx = 2*mapx/(cols-1) -1
mapy = 2*mapy/(rows-1) -1

# 직교좌표를 극좌표로 변환
r, theta = cv2.cartToPolar(mapx, mapy)

# 왜곡 영역만 중심 확대/축소 지수 적용
r[r<scale] = r[r<scale] **exp

# 극좌표를 직교좌표로 변환
mapx, mapy = cv2.polarToCart(r, theta)

# 중심점 기준에서 좌상단 기준으로 변경
mapx = ((mapx+1)*cols -1)/2
mapy = ((mapy+1)*rows -1)/2

# 리매핑 변환
distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow('original', img)
cv2.imshow('distorted', distorted)
cv2.waitKey()
cv2.destroyAllWindows()