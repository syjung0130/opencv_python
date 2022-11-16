# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pylab as plt

'''
2차원 히스토그램과 HSV 컬러 스페이스를 이용하면
색상으로 특정 물체나 사물의 일부분을 배경에서 분리할 수 있다.
기본 원리는 물체가 있는 관심 영역의 H와 V 값의 분포를 얻어낸 후 
전체 영상에서 해당 분포의 픽셀만 찾아내는 것입니다.
'''

win_name = 'back_projection'
img = cv2.imread('../img/pump_horse.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
draw = img.copy()

# 역투영된 결과를 마스킹해서 결과를 출력하는 공통함수
def masking(bp, win_name):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(bp, -1, disc, bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(win_name, result)

# 직접 구현한 역투영 함수
def backProject_manual(hist_roi):
    # 전체 영상에 대한 H, S 히스토그램 계산
    hist_img = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 선택 영역과 전체 영상에 대한 히스토그램 비율 계산
    hist_rate = hist_roi / (hist_img + 1)
    # 비율에 맞는 픽셀 값 매핑
    h, s, v = cv2.split(hsv_img)
    bp = hist_rate[h.ravel(), s.ravel()]
    bp = np.minimum(bp, 1)

