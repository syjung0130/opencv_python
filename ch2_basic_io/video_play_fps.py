# -*- coding: utf-8 -*-
import cv2

# video_file = ".\img\suzzy_mygirl.mp4"
video_file = ".\img\suzzy_mygirl2.mp4"

cap = cv2.VideoCapture(video_file)
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000/fps)
    print("FPS: %f, Delay: %dms" %(fps, delay))

    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow("my girl", img)
            cv2.waitKey(delay)
        else:
            break
else:
    print("can't open video.")

cap.release()
cv2.destroyAllWindows()
