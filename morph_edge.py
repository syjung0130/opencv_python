# -*- coding: utf-8 -*-
import cv2
import numpy as np

'''
# 커널 생성
cv2.getStructuringElement(shape, ksize, anchor)
 - shape: 구조화 커널모양 (cv2.MORPH_RECT, cv2.MORPH_ELIPSE, cv2.MORPH_CROSS)
 - anchor: 구조화 요소의 기준점, cv2.MORPH_CROSS에만 의미있으며 기본 값은 중심점(-1, -1)

# 
dst = cv2.erode(src, kernel, anchor, iteration, borderType, borderValue)
 - src: 입력 영상, 바이너리
 - kernel: 구조화 요소 커널
 - anchor(optional): cv2.getStructuringElement()와 동일
 - iterations(optional): 침식 연산 적용 반복 횟수
 - borderType(optional): 외곽 영역 보정 방법
 - borderValue(optional): 외곽 영역 보정 값
'''

img = cv2.imread('../img/morph_dot.png')

# 구조화 요소 커널, 사각형(3x3) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 침식 연산 적용
erosion = cv2.erode(img, k)

# 결과 출력
merged = np.hstack((img, erosion))
cv2.imshow('Erode', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()