# -*- coding: utf-8 -*-
import numpy as np, cv2
import matplotlib.pylab as plt

# 연산에 사용할 이미지 생성
img1 = np.zeros((200, 400), dtype=np.uint8)
img2 = np.zeros((200, 400), dtype=np.uint8)
img1[:, :200] = 255
img2[100:200, :] = 255

'''
    (img1)           (img2)
+-----+------+   +-----+------+
|     |******|   |************|
|     |******|   +------------|
|     |******|   |            |
+-----+------+   +-----+------+
'''

# 비트와이즈 연산
# and
# 두 이미지 모두 횐색(0이 아닌 부분)만 남는다.. 
'''
+-----+------+
|************|
+-----+******|
|     |******|
+-----+------+
'''
bitAnd = cv2.bitwise_and(img1, img2)

# or
# 두 이미지 모두 0(검은색)인 부분이 white가 되고 나머지는 모두 1(white)이된다.
'''
+-----+------+
|     |******|
|     +------|
|            |
+-----+------+
'''
bitOr = cv2.bitwise_or(img1, img2)


# xor: 두개 모두 다른 값일 경우 1이된다.
# 두 이미지 모두 다른 부분이 모두 1(white)이된다..
'''
+-----+------+
|     |******|
|-----+------|
|*****|      |
+-----+------+
'''
bitXor = cv2.bitwise_xor(img1, img2)
bitNot = cv2.bitwise_not(img1)

# Plot으로 결과 출력
imgs = {'img1': img1, 'img2': img2, 'and': bitAnd,
            'or': bitOr, 'xor': bitXor, 'not(img1)': bitNot}

for i, (title, img) in enumerate(imgs.items()):
    plt.subplot(3, 2, i+1)
    plt.title(title)
    plt.imshow(img, 'gray')
    plt.xticks([]); plt.yticks([])

plt.show()