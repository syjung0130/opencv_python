# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pylab as plt

# 연산에 사용할 이미지 읽기
img1 = cv2.imread('../img/wing_wall.jpg')
img2 = cv2.imread('../img/yate.jpg')

# 이미지 덧셈
img3 = img1 + img2 # 더하기 연산
img4 = cv2.add(img1, img2) # OpenCV 함수


imgs = {'img1':img1, 'img2':img2, 
        'img1+img2':img3, 'cv2.add(img1, img2)':img4}

# 이미지 출력
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(2, 2, i + 1)
    plt.imshow(v[:,:,::-1])
    plt.title(k)
    plt.xticks([]); plt.yticks([])

'''
실행 결과 중 img1+img2는 화소가 고르지 못하고 중간 중간 이상한 색을 띠고 있는 부분이 있는데,
그 부분이 255를 초과한 영역이다.
cv2.add(img1, img2)의 실행 결과는 전체적으로 하얀 픽셀을 많이 가져가므로 좋은 결과로 볼 수 없다.

그래서 좋은 결과를 얻기 위해서는 각 픽셀의 합이 255가 되지 않게 각각의 영상에 가중치를 줘서 계산해야한다.
이 때 각 영상에 적용할 가중치를 알파(alpha) 값이라고 부른다.
'''

plt.show()