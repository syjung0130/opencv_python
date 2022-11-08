#-*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('.\img\sunset.jpg')
x=320; y=150; w=50; h=50 # roi 좌표
roi = img[y:y+h, x:x+w] # roi 지정
# img[150:150+50, 320:320+50], 
# roi 변수에 복사하는 것이 아니다. 참조값들이 저장된다.
# numpy는 x, y를 바꿔서 인덱싱을 해야한다. 
# y를 슬라이싱:행,  x를 슬라이싱:열.

# roi shape, (50, 50, 3)
print("img shape:{0}, roi shape:{1}".format(img.shape, roi.shape))
# void cv::rectangle	(	InputOutputArray 	img,
#     Point 	pt1,
#     Point 	pt2,
#     const Scalar & 	color,
#     int 	thickness = 1,
#     int 	lineType = LINE_8,
#     int 	shift = 0 
# )
# roi 영역에 rectangle을 그린다.
# cv2.rectangle(roi, (0, 0), (h-1, w-1), (0, 255, 0))
# img 영역에 roi rectangle을 그린다.
cv2.rectangle(img, (320, 150), (320+h-1, 150+w-1), (0, 255, 0))
cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
