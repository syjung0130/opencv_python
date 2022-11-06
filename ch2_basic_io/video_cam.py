# -*- coding: utf-8 -*-
import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('camera', img)
            if cv2.waitKey(1) != -1:#1ms동안 입력대기
                break#아무 키나 눌리면 중지
        else:
            print('no frame')
            break
else:
    print("can't open camera.")

cap.release()
cv2.destroyAllWindows()
