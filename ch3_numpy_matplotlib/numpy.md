## Numpy
OpenCV를 사용하다 보면 Numpy배열을 자주 다루게 된다.  
cv2.imread는 Numpy 배열을 반환한다.  
Numpy 배열에서 정보를 얻는 기본 속성은 다음과 같다.
 - ndim: 차원(축)의 수
 - shape: 각 차원의 크기(튜플)
 - size: 전체 요소의 개수, shape의 각 항목의 곱
 - dtype: 요소의 데이터 타입
 - itemsize: 각 요소의 바이트 크기

## 이미지 데이터
blank_500.jpg를 아래처럼 확인해보면 차원이 500, 500, 3인 것을 확인할 수 있다.
~~~python
import cv2
img = cv2.imread('.\img\blank_500.jpg')
type(img)#numpy.ndarray
print(img.ndim)#3
print(img.shape)#(500, 500, 3)
print(img.size)#750000
~~~
width x height인 픽셀 데이터인데,  
한 픽셀당 B, G, R 정보가 있기 때문에  
((width x height) x 3) 차원이 된다. 
img.dtype으로 각 요소의 바이트 크기를 확인해보면 uint8이다.  

## numpy
numpy는 브로드캐스팅 연산, 벡터와 행렬 연산, 푸리에 변환 등  
이미지 프로세싱이나 컴퓨터 비전 분야에서 활용할 수 있는 방대한 기능을 제공한다.  
그래서 OpenCV도움없이도 비전처리를 할수도 있다.  
OpenCV를 쓸 때, OpenCV에 구현되지 않은 연산은 Numpy기능을 이용해서 직접 처리해야하는 경우가 많아서  
기초적인 NumPy 사용법은 반드시 알고 있어야 한다.  


