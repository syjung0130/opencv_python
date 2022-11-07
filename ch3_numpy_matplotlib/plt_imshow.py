# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('.\img\my_girl1.jpg')

plt.imshow(img)
plt.show()
## plt.imgshow함수는 컬러이미지인 경우, R, G, B순으로 해석하지만
## OpenCV 이미지는 B, G, R 순으로 만들어져서 색상의 위치가 반대라서
## 색상이 이상하게 나온다.