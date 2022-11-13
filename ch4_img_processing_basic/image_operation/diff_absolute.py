# -*- coding: utf-8 -*-
import numpy as np, cv2

# 연산에 필요한 영상을 읽고 그레이 스케일로 변환
img1 = cv2.imread('../img/robot_arm1.jpg')
img2 = cv2.imread('../img/robot_arm2.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 두 영상의 절대 값 차 연산
diff = cv2.absdiff(img1_gray, img2_gray)

# 차 영상을 극대화하기 위해 스레시홀드 처리 및 컬러로 변환
# threshold 처리: 차 영상을 하면 1 이상인 부분이 다른 부분이 된다. 
_, diff = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)
# 컬러로 변환
diff_red = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
## row, column, channel(b,g,r)
diff_red[:,:,2] = 0 # red 색상을 표시한다?
# diff_red[:,:,1] = 0 # green 색상으로 표시된다.
# diff_red[:,:,0] = 0 # blue 색상을 표시한다.

# 두번째 이미지에 변화 부분 표시
spot = cv2.bitwise_xor(img2, diff_red)

# 결과 영상 출력
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('diff', diff)
cv2.imshow('spot', spot)
cv2.waitKey()
cv2.destroyAllWindows()