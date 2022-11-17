# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

win_name = 'back_projection'
img = cv2.imread('../img/pump_horse.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
draw = img.copy()

'''
Mat cv::getStructuringElement	(	int 	shape,
Size 	ksize,
Point 	anchor = Point(-1,-1) 
)	

Returns a structuring element of the specified size and shape for morphological operations.

The function constructs and returns the structuring element that can be further passed to erode, dilate or morphologyEx. But you can also construct an arbitrary binary mask yourself and use it as the structuring element.

Parameters
shape	Element shape that could be one of MorphShapes
ksize	Size of the structuring element.
anchor	Anchor position within the element. 
        The default value (−1,−1) means that the anchor is at the center. 
        Note that only the shape of a cross-shaped element depends on the anchor position. 
        In other cases the anchor just regulates how much the result of the morphological operation is shifted.
'''
# 역투영된 결과를 마스킹해서 결과를 출력하는 공통함수
# 스레시홀드와 마스킹을 걸쳐서 결과를 출력.
def masking(bp, win_name):
    # getStructuringElement()와 filter2D()는 마스크의 표면을 부드럽게 하기 위한 것이다.
    # 영상필터 부분에서 더 자세히 보자.
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(bp, -1, disc, bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(win_name, result)

# 직접 구현한 역투영 함수
def backProject_manual(histo_roi):
    # 전체 영상에 대한 H, S 히스토그램 계산
    hist_img = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 선택영역과 전체영상에 대한 히스토그램 비율 계산
    hist_rate = histo_roi / (hist_img + 1)
    # 비율에 맞는 픽셀 값 매핑
    h, s, v = cv2.split(hsv_img)
    bp = hist_rate[h.ravel(), s.ravel()]
    # 비율은 1을 넘어서는 안되기 때문에 1을 넘는 수는 1을 갖게 함.
    bp = np.minimum(bp, 1)
    # 1차원 배열을 원래의 shape으로 변환
    bp = bp.reshape(hsv_img.shape[:2])
    cv2.normalize(bp, bp, 0, 255, cv2.NORM_MINMAX)
    bp = bp.astype(np.uint8)
    # 역투영 결과로 마스킹해서 결과 출력
    masking(bp, 'result_manual')

def backProject_cv(hist_roi):
    # 역투영 함수 호출
    bp = cv2.calcBackProject([hsv_img], [0, 1], hist_roi, [0, 180, 0, 256], 1)
    masking(bp, 'result_cv')

# ROI 선택
(x, y, w, h) = cv2.selectROI(win_name, img, False)
if w > 0 and h > 0:
    roi = draw[y:y+h, x:x+w]
    # 빨간 사각형으로 ROI 영역 표시
    cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 0, 255), 2)#thickness를 2로 설정.
    # 선택한 ROI를 HSV 컬러 스페이스로 변경
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # H, S 채널에 대한 히스토그램 계산
    hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # ROI의 히스토그램을 메뉴얼 구현함수와 OpenCV를 이용하는 함수에 각각 전달.
    backProject_manual(hist_roi)
    backProject_cv(hist_roi)

cv2.imshow(win_name, draw)
cv2.waitKey()
cv2.destroyAllWindows()
    
