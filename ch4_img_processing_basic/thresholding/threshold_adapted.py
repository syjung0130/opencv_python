# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
원본 영상에 조명이 일정하지 않거나 배경색이 여러가지인 경우에는
아무리 여러번 경계값을 바꿔가며 시도해도 
하나의 경계 값을 이미지 전체에 적용해서는 좋은 값을 얻지 못한다.
그래서 이미지를 여러 영역으로 나눈 다음
그 주변 픽셀 값만 가지고 계산을 해서 경계 값을 구해야하는데,
이것을 적응형 스레시홀드(adaptive threshold)라고 한다.
'''
'''
cv2.adaptiveThreshold(img, value, method, type_flag, block_size, C)
 - img: 입력 영상
 - value: 경계 값을 만족하는 픽셀에 적용할 값
 - method: 경계 값 결정 방법
   - cv2.ADAPTIVE_THRESH_MEAN_C: 이웃 픽셀의 평균으로 결정
   - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 가우시안 분포에 따른 가중치의 합으로 결정
 - type_flag: 스레시홀드 적용 방법 지정(cv2.threshold() 함수와 동일)
 - block_size: 영역으로 나눌 이웃의 크기(n x n), 홀수(3, 5, 7, ...)
 - C: 계산된 경계 값 결과에서 가감할 상수(음수 가능)
'''


blk_size = 9 # 블록 사이즈
C = 5 # 차감 상수
img = cv2.imread('../img/sudoku.png', cv2.IMREAD_GRAYSCALE) # 그레이 스케일로 읽기

# 오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, blk_size, C)

th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                        cv2.THRESH_BINARY, blk_size, C)

imgs = {'Original': img, 'Global-Otsu:%d'%ret: th1, \
            'Adapted-Mean': th2, 'Adapted-Gaussian': th3}

for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2, 2, i+1)
    plt.title(k)
    plt.imshow(v, 'gray')
    plt.xticks([]); plt.yticks([])

plt.show()

'''
적응형 스레시홀드를 평균 값과 가우시안 분포를 적용해서 훨씬 좋은 결과가 나온 것을 확인할 수 있다.
가우시안 분포를 이용한 결과는 선명함은 떨어지지만 잡티(noise)가 훨씬 적은 것을 알 수 있다.
'''