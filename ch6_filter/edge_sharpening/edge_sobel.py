# -*- coding: utf-8 -*-
import cv2
import numpy as np

'''
소벨 필터는 중심 픽셀의 차분 비중을 두 배로 준 필터이다.
x축, y축, 대각선 방향의 경계 검출에 모두 강하다.
로버츠 필터와 프리윗 필터는 현재 거의 쓰이지 않는다.
반면에 소벨 필터는 실무적으로도 사용되어서 OpenCV에서 별도의 함수를 제공한다.

dst = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
 - src: 입력영상
 - ddepth: 출력 영상의 dtype(-1: 입력 영상과 동일)
 - dx, dy: 미분 차수(0, 1, 2 중 선택, 둘 다 0일 수는 없음)
 - ksize: 커널의 크기(1, 3, 5, 7 중 선택)
 - scale: 미분에 사용할 계수
 - delta: 연산 결과에 가산할 값
'''

img = cv2.imread('../img/sudoku.jpg')

# 소벨 커널을 직접 생성해서 엣지 검출
# 소벨 커널 생성
gx_k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
gy_k = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# 소벨 필터 적용
edge_gx = cv2.filter2D(img, -1, gx_k)
edge_gy = cv2.filter2D(img, -1, gy_k)

# 소벨 API를 생성해서 엣지 검출
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)

# 결과 출력
merged1 = np.hstack((img, edge_gx, edge_gy, edge_gx+edge_gy))
merged2 = np.hstack((img, sobelx, sobely, sobelx+sobely))
merged = np.vstack((merged1, merged2))

cv2.imshow('sobel', merged)
cv2.waitKey(0)
cv2.destroyAllWindows()