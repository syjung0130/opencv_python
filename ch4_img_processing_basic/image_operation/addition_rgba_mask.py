# -*- coding: utf-8 -*-
import cv2
import numpy as np

'''
배경이 투명한 영상(사진)을 보면,
4개 채널 중 마지막 채널이 배경에 해당하는 영역은 0 값을,
전경에 해당하는 영역은 255 값을 갖는다.
'''

'''
로고 이미지의 네번째 채널이 배경과 전경을 분리할 수 있는 마스크 역할을 해주므로
몇가지 함수의 조합만으로 이미지를 합성할 수 있다.
배경이 투명하므로 배경은 0값을 갖고,
로고 이미지는 0이 아닌 값을 갖는다.
이걸 thresh_binary로 읽어서 255값으로 만든 뒤, 마스킹으로 사용한다.
'''

# 합성에 사용할 영상 읽기, 전경 영상은 4채널 png 파일
img_fg = cv2.imread('../img/opencv_logo.png', cv2.IMREAD_UNCHANGED)
img_bg = cv2.imread('../img/my_girl1.jpg')

# 알파 채널을 이용해서 마스크와 역 마스크 생성
_, mask = cv2.threshold(img_fg[:,:,3], 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# 전경 영상 크기로 배경 영상에서 ROI 잘라내기
img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
h, w = img_fg.shape[:2]
roi = img_bg[10:10+h, 10:10+w]

# 마스크 이용해서 오려내기
masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# 이미지 합성
added = masked_fg + masked_bg
img_bg[10: 10+h, 10:10+w] = added

cv2.imshow('mask', mask)
cv2.resizeWindow('mask', width = w+200, height = h+200)
cv2.imshow('mask_inv', mask_inv)
cv2.resizeWindow('mask_inv', width = w+200, height = h+200)
cv2.imshow('masked_fg', masked_fg)
cv2.resizeWindow('masked_fg', width = w+200, height = h+200)
cv2.imshow('masked_bg', masked_bg)
cv2.resizeWindow('masked_bg', width = w+200, height = h+200)
cv2.imshow('added', added)
cv2.resizeWindow('added', width = w+200, height = h+200)
cv2.imshow('result', img_bg)


cv2.waitKey()
cv2.destroyAllWindows()