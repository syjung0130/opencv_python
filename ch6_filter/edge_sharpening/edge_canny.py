# -*- coding: utf-8 -*-
import cv2, time
import numpy as np

'''
케니 엣지는 다음의 4단계 알고리즘에 따라 경계를 검출한다.
1. 노이즈 제거: 5x5 가우시안 블러링 필터로 노이즈 제거
2. 경계 그레디언트 방향 계산: 소벨 필터로 경계 및 그레디언트 방향 검출
3. 비최대치 억제(Non-Maximum Supression): 그레디언트 방향에서 검출된 경계 중 가장 큰 값만 선택하고 나머지는 제거
4. 이력 스레시홀딩: 두 개의 경계 값(Max, Min)을 지정해서 경계 영역에 있는 픽셀들 중 큰 경계 값(Max) 밖의 픽셀과 연결성이 없는 픽셀 제거

edges = cv2.Canny(img, threshold1, threshold2, edges, apertureSize, L2gradient)
 - img: 입력 영상
 - threshold1, threshold2: 이력 스레시홀딩에 사용할 Min, Max 값
 - apertureSize: 소벨 마스크에 사용할 커널 크기
 - L2gradient: 그레디언트 강도를 구할 방식(True: 제곱 합의 루트, False: 절댓값의 합)
 - edges: 엣지 결과 값을 갖는 2차원 배열
'''

img = cv2.imread('../img/sudoku.jpg')

# 케니 엣지 적용
edges = cv2.Canny(img, 100, 200)

# 결과 출력
cv2.imshow('Original', img)
cv2.imshow('Canny', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()