# -*- coding: utf-8 -*-
import cv2
import numpy as np

'''
x축과 y축의 각 방향으로 차분을 세번 계산해서 경계를 검출하는 필터.
상하 좌우 경계는 뚜렷하게 잘 검출되지만 대각선 검출이 약하다.
'''

img = cv2.imread('../img/sudoku.jpg')

# 프리윗 커널 생성
gx_k = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
gy_k = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# 프리윗 커널 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 결과 출력
merged = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
cv2.imshow('prewitt', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()