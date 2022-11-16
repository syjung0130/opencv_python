# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pylab as plt

'''
 평탄화를 하면 이미지의 밝은 부분이 날아가는 현상이 있다.
 이런 현상을 막기 위해 이미지를 일정한 영역(코드 상의 tileGridSize)으로 나누어
 평탄화를 적용한다.
 이를 피하기 위해 어떤 영역이든 지정된 제한값(코드 상의 clipLimit)을 넘으면
 그 픽셀은 다른 영역에 균일하게 배분하여 적용한다.
 이런 평탄화 방식을 CLAHE라고 한다.
'''

# 이미지를 읽어서 YUV 컬러스페이스로 변경
img = cv2.imread('../img/bright.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 밝기 채널에 대해서 이퀄라이즈 적용
img_eq = img_yuv.copy()
img_eq[:,:,0] = cv2.equalizeHist(img_eq[:,:,0])
img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

# 밝기 채널에 대해서 CLAHE 적용
img_clahe = img_yuv.copy()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #CLAHE 생성
img_clahe[:,:,0] = clahe.apply(img_clahe[:,:,0])
img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)

# 결과 출력
cv2.imshow('Before', img)
cv2.imshow('CLAHE', img_clahe)
cv2.imshow('equalizeHist', img_eq)
cv2.waitKey()
cv2.destroyAllWindows()