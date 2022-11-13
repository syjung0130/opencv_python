# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pylab as plt

# 큐브 영상을 읽어서 HSV로 변환
img = cv2.imread('../img/cube.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 색상별 영역 지정
blue1 = np.array([90, 50, 50])
blue2 = np.array([120, 255, 255])
green1 = np.array([45, 50, 50])
green2 = np.array([75, 255, 255])
red1 = np.array([0, 50, 50])
red2 = np.array([15, 255, 255])
red3 = np.array([165, 50, 50])
red4 = np.array([180, 255, 255])
yellow1 = np.array([20, 50, 50])
yellow2 = np.array([35, 255, 255])