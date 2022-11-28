# -*- coding: utf-8 -*-
import cv2
import numpy as np

file_name = '../img/taekwonv1.jpg'
img = cv2.imread(file_name)

'''
blur() 함수로 블러링
커널의 크기만 지정하면 알아서 평균 커널을 생성해서 블러링 적용한 영상을 만들어 낸다.
이때 커널 크기는 홀수를 사용하는 것이 일반적이다.
cv.blur(src, ksize[, dst[, anchor[, borderType]]]	) ->	dst
 - src: 입력영상
 - ksize: 커널의 크기
'''
blur1 = cv2.blur(img, (10, 10))

'''
normalize 인자에 True를 지정하면 cv2.blur()함수와 같다.
False를 지정하면 커널 영역의 모든 픽셀의 합을 구하게 되는데, 
이것은 밀도를 이용한 객체 추적 알고리즘에서 사용한다.
cv.boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]	) ->	dst
 - src: 입력 영상, NumPy 배열
 - ddepth: 출력 영상의 dtype, -1: 입력 영상과 동일
 - normalize: 커널 크기로 정규화(1/ksize**2) 지정 여부, 불(boolean)
 - 나머지 인자는 filter2D()와 동일
'''
blur2 = cv2.boxFilter(img, -1, (10, 10))

merged = np.hstack((img, blur1, blur2))
cv2.imshow('blur', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()