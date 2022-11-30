# -*- coding: utf-8 -*-
import cv2
import numpy as np

'''
dst = cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst, borderType)
 - src: 입력 영상
 - d : 필터의 직경(diameter), 5보다 크면 매우 느림
 - sigmaColor: 색공간의 시그마 값
 - sigmaSpace: 좌표 공간의 시그마 값
'''

img = cv2.imread('../img/gaussian_noise.jpg')

# 가우시안 필터 적용
blur1 = cv2.GaussianBlur(img, (5, 5), 0)

# 바이레터럴 필터 적용
blur2 = cv2.bilateralFilter(img, 5, 75, 75)

'''
가우시안 필터의 경우, 노이즈를 효과적으로 제거하지만 경계값이 흐릿해진다.
바이레터럴 필터를 사용하면, 노이즈를 제거하면서 경계값도 살릴 수 있다.
'''
merged = np.hstack((img, blur1, blur2))
cv2.imshow('bilateral', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()