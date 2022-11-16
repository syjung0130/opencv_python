# -*- coding: utf-8 -*-
import cv2
import matplotlib.pylab as plt

plt.style.use('classic') # 컬러 스타일을 1.x 스타일로 사용
img = cv2.imread('../img/mountain.jpg')

plt.subplot(131)
# 입력영상: img, 2채널:[0, 1], 마스크할 픽셀:None, 전체, 
#    histSize(계급 갯수): 256으로 하면 색상이 작게 표현되기 때문에 32로 큼직하게 잡음
#    range는 0~256이 두번 반복된다.
hist = cv2.calcHist([img], [0, 1], None, [32, 32], [0, 256, 0, 256])
p = plt.imshow(hist)
plt.title('Blue and Green')
plt.colorbar(p)

plt.subplot(132)
hist = cv2.calcHist([img], [1, 2], None, [32, 32], [0, 256, 0, 256])
p = plt.imshow(hist)
plt.title('Green and Red')
plt.colorbar(p)

plt.subplot(133)
hist = cv2.calcHist([img], [0, 2], None, [256, 256], [0, 256, 0, 256])
p = plt.imshow(hist)
plt.title('Blue and Red')
plt.colorbar(p)

plt.show()