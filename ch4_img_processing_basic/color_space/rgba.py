# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('..\img\opencv_logo.png')
# IMREAD_COLOR 옵션, image가 bgra여도 bgr로 읽어들인다.
# 차원이 row x column x channel: (H, W, 3)이 된다.
bgr = cv2.imread('..\img\opencv_logo.png', cv2.IMREAD_COLOR)#bgr. 
# IMREAD_UNCHANGED 옵션, image가 bgra이면 bgra로 읽어들인다. 차원이 (W, H, 4)가 된다.
# 차원이 row x column x channel: (H, W, 4) 가 된다.
bgra = cv2.imread('..\img\opencv_logo.png', cv2.IMREAD_UNCHANGED)

print("default: ", img.shape, ", color: ", bgr.shape, ", unchanged: ", bgra.shape)

cv2.imshow('bgr', bgr)
cv2.imshow('bgra', bgra)
cv2.imshow('alpha', bgra[:,:,3])
cv2.waitKey(0)
cv2.destroyAllWindows()