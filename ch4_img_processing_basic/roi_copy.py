# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('.\img\sunset.jpg')

x=320; y=150; w=50; h=50
roi = img[y:y+h, x:x+w] # roi 지정
# 열이 x좌표, 행이 x좌표가 된다.
img2 = roi.copy() # roi 배열 복제
# 기존의 roi영역의 오른 쪽에 하나 복제한 데이터를 덮어써서 태양을 2개로 만든다.
img[y:y+h, x+w:x+w+w] = roi

cv2.imshow("img", img)
cv2.imshow("roi", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()