# -*- coding=utf-8 -*-
import cv2

img_file= "./img/my_girl1.jpg"
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

if img is not None:
    cv2.imshow('my girl', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('No image file.')
