# -*- coding=utf-8 -*-
import cv2

img_file= "./img/my_girl_suzzy_221106.jpeg"
save_file = "./img/my_girl_suzzy.jpg"

img = cv2.imread(img_file, cv2.IMREAD_COLOR)
cv2.imshow(img_file, img)
cv2.imwrite(save_file, img)
cv2.waitKey()
cv2.destroyAllWindows()