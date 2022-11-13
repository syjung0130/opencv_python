# -*- coding: utf-8 -*-
import cv2
import numpy as np

alpha = 0.5 # 합성에 사용할 알파 값

# 합성에 사용할 영상 읽기
img1 = cv2.imread('../img/wing_wall.jpg')
img2 = cv2.imread('../img/yate.jpg')

# 수식을 직접 연산해서 알파 블렌딩 적용
blended = img1 * alpha + img2 * (1-alpha)
blended = blended.astype(np.uint8) # 소수점 발생을 제거하기 위함
cv2.imshow('img1 * alpha + img2 * (1 - alpha)', blended)

'''
cv2.addWeight(img1, alpha, img2, beta, gamma)
 - img1, img2: 합성할 두 영상
 - alpha: img1에 지정할 가중치(알파 값)
 - beta: img2에 지정할 가중치, 흔히(1-alpha) 적용
 - gamma: 연산 결과에 가감할 상수, 흔히 0(zero) 적용
'''
# addWeighted() 함수로 알파블렌딩 적용
dst = cv2.addWeighted(img1, alpha, img2, (1-alpha), 0)
cv2.imshow('cv2.addWeighted', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()