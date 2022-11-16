# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pylab as plt

# 이미지 그레이 스케일로 읽기 및 출력
img = cv2.imread('../img/mountain.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)

'''
## 용어 정리
 - BINS: 히스토그램 그래프의 X축의 간격이다.
  0 ~ 255를 표현하였기 때문에 BINS값은 256이 된다.
  BINS값이 16이면 0~15, 16~31, ...,240~255와 같이 X축이 16개로 표현된다.
 - DIMS: 이미지에서 조사하고자 하는 값을 의미한다.
  빛의 강도를 조사할 것인지, RGB값을 조사할 것인지를 결정한다.
 - RANGE: 측정하고자 하는 값의 범위.
  X축의 from ~ to로 이해할 수 있다.

cv2.calcHist(images, channels, mask, histSize, ranges [,hist [, accynulate]])
  - image: 분석대상 이미지(uint8 or float32 type). Array형태.
  - channels: 분석 채널(X축의 대상).
   이미지가 gray scale이면 [0], color 이미지이면 [0], [0, 1]형태
   (1: Blue, 2: Gree, 3: Red)
  - mask: 이미지의 분석영역. None이면 전체 영역.
  - histSize - BINS값[256]
  - range: Range값. [0, 256]
'''
#channel: 0(그레이스케일), mask: None(전체영역), histSize: 256(0~255), range: [0, 256]
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)

print('hist.shape: ', hist.shape) # 히스토그램의 shape (256, 1)
print('hist.sum(): ', hist.sum(), ', img.shape: ', img.shape)
print('hist: {0}'.format(hist))
plt.show()