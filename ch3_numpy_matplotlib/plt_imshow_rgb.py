# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('.\img\my_girl1.jpg')

plt.imshow(img[:,:,::-1])
## 컬러 채널의 순서를 바꾼다.
## 3차원 배열의 모든 내용을 선택하는 것은 img[:,:,:] 이다.  
## 마지막 축의 길이가 3이므로 다시 img[:,:,::]로 바꾸어 쓸 수 있다.
## 이 때, 마지막 축의 요소의 숮서를 거꾸로 뒤집기 위해 img[:,:,::-1]로 쓸 수 있다.
## 이걸 풀어서 쓰면 아래와 같다.
## img[:,:,(2,1,0)]
## 또는
## img[:,:,2], img[:,:,1], img[:,:,0] = img[:,:,0], img[:,:,1], img[:,:,2] 


plt.xticks([])
plt.yticks([])
plt.show()
## plt.imgshow함수는 컬러이미지인 경우, R, G, B순으로 해석하지만
## OpenCV 이미지는 B, G, R 순으로 만들어져서 색상의 위치가 반대라서
## 색상이 이상하게 나온다.

