# -*- coding: utf-8 -*-
import cv2
import numpy as np

isDragging = False # 마우스 드래그 상태 저장
x0, y0, w, h = -1, -1, -1, -1 # 영역 선택 좌표 저장