# -*- coding: utf-8 -*-
import numpy as np, cv2

# 이미지 읽기, BGR스케일
img = cv2.imread('../img/yate.jpg')

# 컬러 스케일을 BGR에서 YUV로 변경
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# BGR을 YUV나 HSV로 변경하면, 하나의 밝기 채널만 조절하면 된다.
# YUV 컬러 스케일의 첫번째 채널에 대해서 이퀄라이즈 적용
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

# 컬러 스케일을 YUV에서 BGR로 변경
img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow('Before', img)
cv2.imshow('After', img2)

cv2.waitKey()
cv2.destroyAllWindows()