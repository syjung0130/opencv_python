# -*- coding: utf-8 -*-
import cv2
import numpy as np

'''
원근 변환(perspective transform)은 보는 사람의 시각에 따라 같은 물체도 먼것은 작게,
가까운 것은 크게 보이는 현상인 원근감을 주는 변환을 말한다.
우리가 원근감을 느끼는 이유는 실제세계가 3차원 좌표계이기 때문인데, 영상은 2차원 좌표계이다. 
그래서 차원간의 차이를 보정해줄 추가 연산과 시스템이 필요한데,
이 때 사용하는 좌표계를 동차 좌표(homogeneous coordinates)라고 한다.

OpenCV는 변환 전과 후를 짝짓는 4개의 매칭 좌표만 지정해주면 
원근 변환에 필요한 3x3 변환행렬을 계산해주는 cv2.getPerspectiveTransform() 함수를 제공한다.
 - mtrx = cv2.getPerspectiveTransform(pts1, pts2)
   - pts1: 변환 이전 영상의 좌표 4개, 4x2 NumPy 배열(float32)
   - pts2: 변환 이후 영상의 좌표 4개, pts1과 동일
   - mtrx: 변환행렬 반환, 3x3 행렬

원근 변환을 위해서는 cv2.warpPerspective()함수를 사용하면 된다.
 - dst = cv2.warpPerspective(src, mtrx, dsize, [, dst, flags, borderMode, borderValue])
'''

file_name = "../img/fish.jpg"
img = cv2.imread(file_name)
rows, cols = img.shape[0:2]

# 원근 변환 전후 4개 좌표
pts1 = np.float32([[0,0], [0,rows], [cols,0], [cols, rows]])
pts2 = np.float32([[100, 50], [10, rows-50], [cols-100, 50], [cols-10, rows-50]])

# 변환 전 좌표를 원본 이미지에 표시
cv2.circle(img, (0,0), 10, (255, 0, 0), -1)
cv2.circle(img, (0,rows), 10, (0, 255, 0), -1)
cv2.circle(img, (cols,0), 10, (0, 0, 255), -1)
cv2.circle(img, (cols,rows), 10, (0, 255, 255), -1)

# 원근 변환 행렬 계산
mtrx = cv2.getPerspectiveTransform(pts1, pts2)

# 원근 변환 적용
dst = cv2.warpPerspective(img, mtrx, (cols, rows))

cv2.imshow('origin', img)
cv2.imshow('perspective', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()