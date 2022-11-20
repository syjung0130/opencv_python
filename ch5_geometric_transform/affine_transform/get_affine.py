# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt


'''
어핀 변환은 이동, 확대/축소, 회전을 포함하는 변환으로
직선, 길이의 비율, 평행성을 보존하는 변환을 말한다.
어핀변환의 이런 성질 때문에 변환 전과 후의 3개의 점을 짝지어 매핑할 수 있다면 
변환행렬을 거꾸로 계산할 수 있다.
'''

file_name = '../img/fish.jpg'
img = cv2.imread(file_name)
rows, cols = img.shape[:2]

# 변환 전, 후 각 3개의 좌표 생성
pts1 = np.float32([[100, 50], [200, 50], [100, 200]])
pts2 = np.float32([[80, 70], [210, 60], [250, 120]])

# 변환 전 좌표를 이미지에 표시
cv2.circle(img, (100, 50), 5, (255, 0), -1)
cv2.circle(img, (200, 50), 5, (0,255, 0), -1)
cv2.circle(img, (100, 200), 5, (0, 0,255), -1)

'''
변환 전과 후의 좌표만 작성하면 변환행렬을 getAffineTransform()함수로 계산한다.
그리고 나서 변환행렬을 warpAffine()에 넘기면 변환이 된다.,
'''

# 짝지은 3개의 좌표로 변환행렬 계산
mtrx = cv2.getAffineTransform(pts1, pts2)
# 어핀 변환 적용
dst = cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))

# 결과 출력
cv2.imshow('origin', img)
cv2.imshow('affin', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()