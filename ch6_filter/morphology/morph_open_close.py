# -*- coding: utf-8 -*-
import cv2
import numpy as np

'''
침식은 어두운 부분의 노이즈를 제거하는 효과가 있고
팽창은 밝은 부분의 노이즈를 제거하는 효과가 있다.
한 필터만 사용하면 원래 모양이 홀쭉해지거나 뚱뚱해지는 변형이 일어난다.
그래서 침식과 팽창의 연산을 조합하면 원래의 모양을 유지하면서 노이즈를 제거할 수 있다.

열림: 침식+팽창
주변보다 밝은 노이즈를 제거하는데 효과적이다. 
맞닿아 있는 것처럼 보이는 독립된 개체를 분리하거나 돌출된 모양을 제거하는데 효과적이다.

닫힘: 팽창+침식
주변보다 어두운 노이즈를 제거하는데 효과적이다.
끊어져 보이는 개체를 연결하거나 구멍을 메우는데 효과적이다.

그레디언트: 팽창-침식
팽창연산을 적용한 이미지에서 침식연산을 적용한 이미지를 빼면 경계 픽셀만 얻게 된다.
이는 경계 검출과 비슷한데, 이를 그레디언트(gradient)연산이라고 한다.

탑햇: 원본 - 열림
원본에서 열림 연산 적용 결과를 빼면 값이 크게 튀는 밝은 영역을 강조할 수 있다.

블랙햇: 닫힘 - 원본
닫힘 연산 적용 결과에서 원본을 빼면 어두운 부분을 강조할 수 있다.
'''

img1 = cv2.imread('../img/morph_dot.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('../img/morph_hole.png', cv2.IMREAD_GRAYSCALE)

# 구조화 요소 커널, 사각형(5x5) 생성
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 열림 연산 적용
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, k)
# 닫힘 연산 적용
closing = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, k)

# 결과 출력
merged1 = np.hstack((img1, opening))
merged2 = np.hstack((img2, closing))
merged3 = np.vstack((merged1, merged2))
cv2.imshow('opening, closing', merged3)
cv2.waitKey(0)
cv2.destroyAllWindows()