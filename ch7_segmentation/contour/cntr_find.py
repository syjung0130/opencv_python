# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('../img/shapes.png')
img2 = img.copy()

# 그레이 스케일로 변환
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 스레시홀드로 바이너리 이미지로 만들어서 검은색 배경에 흰색 전경으로 반전
ret, imthres = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY_INV)

# 가장 바깥쪽 컨투어에 대해 모든 좌표 반환
## cv2.RETR_EXTERNAL(바깥쪽 컨투어), cv2.CHAIN_APPROX_NONE(모든 좌표 반환)
contour, hierachy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, \
                                                cv2.CHAIN_APPROX_NONE)[-2:]

# 가장 바깥쪽 컨투어에 대해 꼭짓점 좌표만 반환
contour2, hierachy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, \
                                                cv2.CHAIN_APPROX_SIMPLE)[-2:]

# 각 컨투어의 개수 출력
print('도형의 개수: %d(%d)'% (len(contour), len(contour2)))

# 모든 좌표를 갖는 컨투어 그리기, 초록색
cv2.drawContours(img, contour, -1, (0, 255, 0), 4)
# 꼭짓점 좌표만을 갖는 컨투어 그리기, 초록색
cv2.drawContours(img, contour2, -1, (0, 255, 0), 4)

# 컨투어의 모든 좌표를 작은 파란색 점(원)으로 표시
for i in contour:
    for j in i:
        cv2.circle(img, tuple(j[0]), 1, (255, 0, 0), -1)

# 컨투어의 모든 좌표를 작은 파란색 점(원)으로 표시
for i in contour2:
    for j in i:
        cv2.circle(img2, tuple(j[0]), 1, (255, 0, 0), -1)

# 결과 출력
cv2.imshow('CHAIN_APPROX_NONE', img)
cv2.imshow('CHAIN_APPROX_SIMPLE', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()