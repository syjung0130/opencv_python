import cv2
import numpy as np
import matplotlib.pylab as plt

'''
종이에 출력된 문서를 바이너리 이미지로 만들 경우,
 - 새하얀 종이에 검은 잉크로 출력된 문서의 영상이라면 스레시홀드가 굳이 필요하지 않다.
 - 하지만, 현실에서는 흰색, 누런색, 회색 종이에 검은색, 파란색으로 인쇄된 문서가 더 많다.
 - 그래서 적절한 경계값을 정하기 위해서는 여러 차례에 걸쳐 경계 값을 조금씩 수정해가면서
   가장 좋은 경계 값을 찾아야한다.
 - threshold값을 80부터 20씩 증가시키면서 확인해보면 120 ~ 140 사이에 좋은 결과를 얻을 수 있는 걸 볼 수있다.
 - 오츠 노부유키는 이렇게 반복적인 시도 없이 한번에 효율적으로 경계 값을 찾을 수 있는 방법을 제안했고
   그 이름을 따서 오츠의 이진화 알고리즘이라고 부른다.
 - OpenCV에서는 threshold의 flag로 cv2.THRESH_OTSU를 전달해서 사용할 수 있다.
'''

# 이미지를 그레이 스케일로 읽기
img = cv2.imread('../img/scaned_paper.jpg', cv2.IMREAD_GRAYSCALE)
# 경계 값을 130으로 지정
_, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
# 경계 값을 지정하지 않고 오츠의 알고리즘 선택
t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('otsu threshold: ', t)

imgs = {'Original': img, 't:130': t_130, 'otsu:%d'%t: t_otsu}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3, i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([]); plt.yticks([])

plt.show()