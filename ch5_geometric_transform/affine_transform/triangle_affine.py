# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('../img/taekwonv1.jpg')
img2 = img.copy()
draw = img.copy()

'''
1.변환 전 삼각형 좌표 3쌍을 정한다
2.변환 후 삼각형 좌표 3쌍을 정한다
3.과정 1의 삼각형 좌표를 완전히 감싸는 외접 사각형 좌표를 구한다.
4.과정 3의 사각형 영역을 관심 영역으로 지정한다.
5.과정 4의 관심영역을 대상으로 과정 1과 과정 2의 좌표로 변환행렬을 구하여 어핀 변환한다.
6.과정 5의 변환된 관심영역에서 과정 2의 삼각형 좌표만 마스킹한다.
7.과정 6의 마스크를 이용해서 원본 또는 다른 영상에 합성한다.

과정 3에서 삼각형 뿐 아니라 다각형의 좌표를 전달하면 정확히 감싸는 외접 사각형의 좌표를 반환한다.
x,y,w,h = cv2.boundingRect(pts)
 - pts: 다각형의 좌표
 - x, y, w, h: 외접 사각형의 좌표와 폭과 높이

과정 6에서 삼각형 마스크를 생성하기 위해 cv2.fillConvexPoly()함수를 쓰면 편리하다.
cv2.fillConvexPoly(img, points, color [, lineTypes])
 - img: 입력 영상
 - points: 다각형 꼭짓점 좌표
 - color: 채우기에 사용할 색상
 - lineType: 선 그리기 알고리즘 선택 플래그
'''

# 변환 전, 후 삼각형 좌표
pts1 = np.float32([[188, 14], [85, 202], [294, 216]])
pts2 = np.float32([[128, 40], [85, 307], [306, 167]])

# 각 삼각형을 완전히 감싸는 사각형 좌표 구하기
x1, y1, w1, h1 = cv2.boundingRect(pts1)
x2, y2, w2, h2 = cv2.boundingRect(pts2)

# 사각형을 이용한 관심영역 설정
roi1 = img[y1:y1+h1, x1:x1+w1]
roi2 = img2[y2:y2+h2, x2:x2+w2]

# 관심영역을 기준으로 좌표 계산
offset1 = np.zeros((3, 2), dtype=np.float32)
offset2 = np.zeros((3, 2), dtype=np.float32)
for i in range(3):
    offset1[i][0], offset1[i][1] = pts1[i][0] - x1, pts1[i][1] - y1
    offset2[i][0], offset2[i][1] = pts2[i][0] - x2, pts2[i][1] - y2

# 관심 영역을 주어진 삼각형 좌표로 어핀 변환
mtrx = cv2.getAffineTransform(offset1, offset2)
warped = cv2.warpAffine(roi1, mtrx, (w2, h2), None, \
                            cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)

# 어핀 변환 후 삼각형만 골라내기 위한 마스크 생성
mask = np.zeros((h2, w2), dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32(offset2), (255))

# 삼각형 영역만 마스킹해서 합성
warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
roi2_masked = cv2.bitwise_and(roi2, roi2, mask=cv2.bitwise_not(mask))
roi2_masked = roi2_masked + warped_masked
img2[y2:y2+h2, x2:x2+w2] = roi2_masked

# 관심영역과 삼각형에 선을 그려서 출력
cv2.rectangle(draw, (x1, y1), (x1+w1, y1+h1), (0, 255), 1)
cv2.polylines(draw, [pts1.astype(np.int32)], True, (255, 0, 0), 1)
cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 1)
cv2.imshow('origin', draw)
cv2.imshow('warped triangle', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
