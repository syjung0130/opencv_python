# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('../img/taekwonv1.jpg')
rows, cols = img.shape[:2]

# 마스크 생성, 원래 이미지보다 2픽셀 크게
mask = np.zeros((rows+2, cols+2), np.uint8)

# 채우기에 사용할 색
newVal = (255, 255, 255)

loDiff, upDiff = (10, 10, 10), (10, 10, 10)

# 마우스 이벤트 처리 함수
def onMouse(event, x, y, flags, param):
    global mask, img
    if event == cv2.EVENT_LBUTTONDOWN:
        seed = (x, y)
        # 색 채우기 적용
        retval = cv2.floodFill(img, mask, seed, newVal, loDiff, upDiff)
        cv2.imshow('img', img)

# 화면 출력
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()