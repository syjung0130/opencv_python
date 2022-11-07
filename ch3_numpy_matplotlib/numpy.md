## Numpy
OpenCV를 사용하다 보면 Numpy배열을 자주 다루게 된다.  
cv2.imread는 Numpy 배열을 반환한다.  
Numpy 배열에서 정보를 얻는 기본 속성은 다음과 같다.
 - ndim: 차원(축)의 수
 - shape: 각 차원의 크기(튜플)
 - size: 전체 요소의 개수, shape의 각 항목의 곱
 - dtype: 요소의 데이터 타입
 - itemsize: 각 요소의 바이트 크기

## 1. 이미지 데이터와 NumPy
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

numpy는 브로드캐스팅 연산, 벡터와 행렬 연산, 푸리에 변환 등  
이미지 프로세싱이나 컴퓨터 비전 분야에서 활용할 수 있는 방대한 기능을 제공한다.  
그래서 OpenCV도움없이도 비전처리를 할수도 있다.  
OpenCV를 쓸 때, OpenCV에 구현되지 않은 연산은 Numpy기능을 이용해서 직접 처리해야하는 경우가 많아서  
기초적인 NumPy 사용법은 반드시 알고 있어야 한다.  

## 2. NumPy 배열 생성
 - 값으로 생성: array()
 - 초기 값으로 생성: empty(), zeros(), ones(), full()
 - 기존 배열로 생성: empty_like(), zeros_like(), ones_like(), full_like()
 - 순차적인 값으로 생성: arrange()
 - 난수로 생성: random.rand(), random.randn()

## 3. 값으로 생성
배열 생성에 사용할 값을 가지고 있는 경우에는 numpy.array()함수로 간단히 생성할 수 있다.  
 - numpy.array(list, [, dtype]): 지정한 값들로 NumPy 배열 생성
   - list: 배열 생성에 사용할 값을 갖는 파이썬 리스트 객체
   - dtype: 데이터 타입(생략하면 값에 의해 자동 결정)
     - int8, int16, int32, int64: 부호 있는 정수
     - uint8, uint16, uint32, uint64: 부호 없는 정수
     - float16, float32, float64, float128: 부동 소수점을 갖는 실수
     - complex64, complex128, complex256: 부동 소수점을 갖는 복소수
     - bool: 불(boolean)

~~~python
## array() 배열 생성이 가능하다.
>>> import numpy as np
>>> a = np.array([1, 2, 3, 4])
>>> a
array([1, 2, 3, 4])
>>> a.dtype
dtype('int32')
>>> a.shape
(4,)
>>>

## 다차원 배열도 생성이 가능하다.
>>> b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
>>> b
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])
>>> b.shape
(2, 4)
>>>

## 타입이 자동으로 결정된다.
>>> c = np.array([1, 2, 3.14, 4])
>>> c
array([1.  , 2.  , 3.14, 4.  ])
>>> c.dtype
dtype('float64')

## 타입을 직접 지정할 수도 있다.
>>> d = np.array([1, 2, 3, 4], dtype=np.float32)
>>> d
array([1., 2., 3., 4.], dtype=float32)
>>>
~~~

## 4. 크기와 초기값으로 생성
 - numpy.empty(shape, [, dtype]): 초기화되지 않은 값(쓰레기 값)으로 배열 생성
   - shape: 튜플, 배열의 각 차수의 크기 지정
 - numpy.zeros(shape, [, dtype]): 0으로 초기화된 배열 생성
 - numpy.ones(shape, [, dtype]): 1로 초기화된 배열 생성
 - numpy.full(shape, fill_value [, dtype]): fill_value로 초기화된 배열 생성

~~~python
>>> a=np.empty((2, 3))
>>> a
array([[1.78019761e-306, 4.45049997e-308, 9.34598926e-307],
       [1.33511562e-306, 6.23040373e-307, 1.60219035e-306]])
>>> a.dtype
dtype('float64')

## fill함수로 배열의 모든 원소들을 value로 채울 수 있다. 
>>> a.fill(255)
>>> a
array([[255., 255., 255.],
       [255., 255., 255.]])

## 배열을 생성하고 특정한 값으로 초기화하는 함수들 - zeros()
>>> b = np.zeros((2, 3))
>>> b
array([[0., 0., 0.],
       [0., 0., 0.]])
>>> b.dtype
dtype('float64')

## 위에서는 자동으로 float64로 타입이 지정되었지만 zeros()함수도 타입을 직접 지정할 수 있다.
>>> c = np.zeros((2, 3), dtype=np.int8)
>>> c
array([[0, 0, 0],
       [0, 0, 0]], dtype=int8)

## ones()함수를 사용하면 모든 원소들의 값을 1로 초기화할 수 있다.
>>> d = np.ones((2, 3), dtype=np.int16) 
>>> d
array([[1, 1, 1],
       [1, 1, 1]], dtype=int16)

## full()함수를 사용하면 모든 원소들의 값을 지정한 값(255)으로 생성과 초기화가 가능하다.
>>> e = np.full((2, 3, 4), 255, dtype=np.uint8)
>>> e
array([[[255, 255, 255, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255]],

       [[255, 255, 255, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255]]], dtype=uint8)

>>> import cv2
>>> img = cv2.imread('.\img\my_girl1.jpg')
>>> img
array([[[108,  82, 122],
        [108,  82, 122],
        [108,  82, 122],
        ...,
        ...,
        [173, 157, 144],
        [174, 158, 145],
        [175, 159, 146]]], dtype=uint8)
>>>
>>> img.shape
(1199, 791, 3)

>>> import numpy as np
>>> a = np.empty_like(img)
>>> b = np.zeros_like(img)
>>> c = np.ones_like(img)
>>> d = np.full_like(img, 255)

>>> a
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,

       ...,
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype=uint8)
>>> a.shape
(1199, 791, 3)

>>> b
array([[[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        ...,
        ...,
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]], dtype=uint8)

>>> c
array([[[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        ...,
        ...,
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]]], dtype=uint8)

>>> d
array([[[255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        ...,
        ...,
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]]], dtype=uint8)

>>> a.shape
(1199, 791, 3)
>>> b.shape
(1199, 791, 3)
>>> c.shape
(1199, 791, 3)
>>> d.shape
(1199, 791, 3)
~~~

## 5. 시퀀스와 난수로 생성
NumPy 배열을 생성하는 방법 중에는 일정한 범위 내에서 순차적인 값을 갖게하는 방법과  
난수로 채우는 방법이 있다.  
 - numpy.arrange([start=0, ] stop [, step=1, dtype=float64]): 순차적인 값으로 생성
   - start: 시작 값
   - stop: 종료 값, 범위에 포함하는 수는 stop -1까지
   - step: 증가 값
   - dtype: 데이터 타입
 - numpy.random.rand([d0 [, d1 [..., dn]]]): 0과 1 사이의 무작위 수로 생성
   - d0, d1..dn: shape, 생략하면 난수 한 개 반환
 - numpy.random.randn([d0 [, d1 [..., dn]]]): 표준정규 분포(평균:0, 분산:1)를 따르는 무작위 수로 생성

~~~python
>>> import numpy as np
# range()와 비슷하지만, arange()함수는 리스트가 아닌 NumPy배열을 반환한다. 
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a.dtype
dtype('int32')
>>> a.shape
(5,)

# 인자로 소수점이 있는 수를 사용하면 dtype은 float64로 지정되며, 필요에 따라 명시적으로 지정할 수 있다.
>>> b = np.arange(5.0)
>>> b
array([0., 1., 2., 3., 4.])
>>> b.dtype
dtype('float64')

# 3에서 시작해서 9의 바로 앞 그러니까 8까지 2씩 증가하는 수를 갖는 배열을 생성
>>> c = np.arange(3, 9, 2) 
>>> c
array([3, 5, 7])
~~~
  
난수를 발생하는 함수로는 random.rand()와 random.randn()이 있다.  
rand()함수는 0과 1 사이의 값을 무작위로 만들고,  
randn()함수는 평균이 0이고 분산이 1인 정규 분포를 따르는 무작위 수를 만들어낸다.  
~~~python
>>> np.random.rand()
0.1560600113136057
-0.03497220924026316
>>>
>>> a = np.random.rand(2, 3)
>>> a
array([[0.35078328, 0.15527345, 0.92812208],
       [0.5925493 , 0.17662502, 0.15798291]])
>>> b = np.random.randn(2,3)
>>> b
array([[ 1.63397415, -0.11264518, -1.02838885],
       [-0.27655199, -0.77738498,  0.94096311]])
~~~

## 6. dtype 변경
배열의 데이터 타입을 변경하는 함수들.  
 - ndarray.astype(dtype)
   - dtype: 변경하고 싶은 dtype, 문자열 또는 dtype
 - numpy.uintXX(array): array를 부호 없는 정수(uint) 타입으로 변경해서 반환
   - uintXX: uint8, uint16, uint32, uint64
 - numpy.intXX(array): array를 int 타입으로 변경해서 반환
   - intXX: int8, int16, int32, int64
 - numpy.floatXX(array): array를 float타입으로 변경해서 반환
   - floatXX: float16, float32, float64, float128
 - numpy.complexXX(array): array를 복소수(complex)타입으로 변경해서 반환
   - complexXX: complex64, complex128, complex256
~~~python
>>> import numpy as np
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a.dtype
dtype('int32')
>>> b = a.astype('float32')
>>> b
array([0., 1., 2., 3., 4.], dtype=float32)
~~~
배열 객체의 dtype을 변경하는 방법으로 배열 객체의 astype()메서드를 호출하는 방법이 있는 반면 또 다른 방법도 있다.  
NumPy 모듈 정적함수에는 NumPy에서 지원하는 dtype들과 같은 이름의 함수들이 있는데,  
이 함수들 중에 변경을 원하는 dtype 이름의 함수를 호출하면서 배열 객체를 인자로 전달하는 방법도 있다.  
~~~python
>>> a.dtype
dtype('int32')
>>> d = np.uint8(a)
>>> d
array([0, 1, 2, 3, 4], dtype=uint8)
~~~

## 7. 차원 변경
 - ndarray.reshape(newshape): ndarray의 shape를 newshape로 차원 변경
 - numpy.reshape(ndarray, newshape): ndarray의 shape를 newshape로 차원 변경
   - ndarray: 원본 배열 객체
   - newshape: 변경하고자하는 새로운 shape(튜플)
 - numpy.ravel(ndarray): 1차원 배열로 차원 변경
   - ndarray: 변경할 원본 배열
 - ndarray.T: 전치배열(transpose)
원래는 1차원이던 배열을 2행 3열 배열로 바꾼다든지,  
100x200x3인 배열을 1차원으로 바꾸는 식의 작업이 필요할 때가 많다.  
원본 배열의 메서드로 호출하거나  
NumPy모듈에 있는 정적함수에 배열을 인자로 전달해서 호출한다.  
그때 그때 편리한 방법을 사용하면 된다.
~~~python
>>> import numpy as np
# 배열 생성
>>> a = np.arange(6)
>>> a
array([0, 1, 2, 3, 4, 5])
# 원본 배열의 메서드로 reshape메서드 호출
>>> b = a.reshape(2,3)
>>> b
array([[0, 1, 2],
       [3, 4, 5]])
# NumPy모듈의 정적함수에 배열을 인자로 전달해서 호출
>>> c = np.reshape(a, (2, 3))
>>> c
array([[0, 1, 2],
       [3, 4, 5]])
~~~

두 함수 모두 새로운 shape를 지정할 때 -1을 포함해서 전달할 수 있다.  
-1의 의미는 해당 차수에 대해서는 크기를 지정하지 않겠다는 뜻이다.  
그것은 나머지 차수를 이용해서 알아서 계산해달라는 뜻이다.  
~~~python
>>> d = np.arange(100).reshape(2, -1) 
>>> d
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49],
       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
        98, 99]])

>>> e = np.arange(100).reshape(-1, 5) 
>>> e
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24],
       [25, 26, 27, 28, 29],
       [30, 31, 32, 33, 34],
       [35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44],
       [45, 46, 47, 48, 49],
       [50, 51, 52, 53, 54],
       [55, 56, 57, 58, 59],
       [60, 61, 62, 63, 64],
       [65, 66, 67, 68, 69],
       [70, 71, 72, 73, 74],
       [75, 76, 77, 78, 79],
       [80, 81, 82, 83, 84],
       [85, 86, 87, 88, 89],
       [90, 91, 92, 93, 94],
       [95, 96, 97, 98, 99]])
>>> e.shape
(20, 5)
~~~
두번째 예를 살펴보자.  
(-1, 5)는 -1행 5열을 생성하겠다는 것인데, 5열에 맞춰서 행을 알아서 계산하면  
20행이 나오므로 (20, 5)가 출력됩니다. -1은 개발자에게 불필요한 계산을 하지 않아도 되게 해주므로 편리하다.  
하지만 101개의 1차원 배열을 2열로 나누라는 식의 연산은 오류가 발생하므로 주의해야한다.  
  
어떤 배열을 1차원 배열로 재정렬할 수 있는 방법은 방금 설명한 reshape()함수를 이용해도되고  
numpy.ravel()함수를 사용해도 된다.  
-1을 사용하면 더 간단히 사용할 수 있다.
~~~python
>>> f = np.zeros((2, 3)) 
>>> f
array([[0., 0., 0.],
       [0., 0., 0.]])
>>> f.reshape((6,)) 
array([0., 0., 0., 0., 0., 0.])
>>> f.reshape(-1) 
array([0., 0., 0., 0., 0., 0.])
>>> np.ravel(f)
array([0., 0., 0., 0., 0., 0.])
~~~

NumPy배열(ndarray) 객체에는 ndarray.T라는 속성이 있다.  
이 속성을 이용하면 행과 열을 서로 바꾸는 전치배열을 얻을 수 있다.  
~~~python
>>> g = np.arange(10).reshape(2, -1) 
>>> g
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])
>>> g.T
array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [4, 9]])
~~~

## 8. 브로딩캐스팅 연산
브로드캐스팅이 지원되지 않으면 아래와 같이 for문을 사용해서 일일히 연산을 해주어야한다.
~~~python
>>> import numpy as np
>>> mylist = list(range(10))
>>> mylist
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> for i in range(len(mylist)):
...     mylist[i] = mylist[i]+1
... 
>>> mylist
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
~~~
브로드캐스팅 연산을 사용하면 아래처럼 쉽게 연산이 가능하다.
~~~python
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> a+1
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
~~~

브로드캐스팅 연산은 더하기 연산 뿐만 아니라 모든 산술 연산이 가능하다.  
아래 코드는 NumPy배열과 스칼라(Scalar, 스케일러) 값 간의 여러가지 연산의 예를 보여주고 있다.  
~~~python
>>> import numpy as np
>>> a = np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> a+5
array([5, 6, 7, 8, 9])
>>> a-2
array([-2, -1,  0,  1,  2])
>>> a*2
array([0, 2, 4, 6, 8])
>>> a/2
array([0. , 0.5, 1. , 1.5, 2. ])
>>> a**2
array([ 0,  1,  4,  9, 16], dtype=int32)
>>> b = np.arange(6).reshape(2, -1)
>>> b
array([[0, 1, 2],
       [3, 4, 5]])
>>> b*2
array([[ 0,  2,  4],
       [ 6,  8, 10]])
~~~
산술 연산 뿐만 아니라 비교 연산도 가능하다.  
~~~python
>>> a
array([0, 1, 2, 3, 4])
>>> a > 2
array([False, False, False,  True,  True])
~~~

배열과 숫자 값 간의 연산 뿐만 아니라 배열끼리의 연산도 가능하다.  
~~~python
>>> import numpy as np
>>> a = np.arange(10, 60, 10)
>>> b = np.arange(1, 6)
>>> a
array([10, 20, 30, 40, 50])
>>> b
array([1, 2, 3, 4, 5])
>>> a+b
array([11, 22, 33, 44, 55])
>>> a-b
array([ 9, 18, 27, 36, 45])
>>> a*b
array([ 10,  40,  90, 160, 250])
>>> a/b
array([10., 10., 10., 10., 10.])
>>> a**b
array([       10,       400,     27000,   2560000, 312500000], dtype=int32)
~~~

하지만, 배열 간의 연산에는 약간의 제약이 있다.
두 배열의 shape가 완전히 동일하거나  
둘 중 하나가 1차원이면서 1차원 배열의 축의 길이가 같아야한다.  
~~~python
>>> import numpy as np
>>> a = np.ones((2,3))
>>> b = np.ones((3, 2))
>>> a + b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (2,3) (3,2)
~~~
두 배열의 shape가 일치하지 않아서 연산에 실패한다.  
  
배열 c가 1차원이고 1차원 배열의 열의 개수가 a배열의 열의 갯수와 같아서 연산이 가능하다.
~~~python
>>> import numpy as np
>>> a = np.ones((2,3))
>>> a
array([[1., 1., 1.],
       [1., 1., 1.]])
>>> c = np.arange(3)
>>> c
array([0, 1, 2])
>>> a+c
array([[1., 2., 3.],
       [1., 2., 3.]])
~~~

열의 갯수가 맞지 않으면 계산에 실패한다.
~~~python
>>> d = np.arange(2)
>>> a + d
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: operands could not be broadcast together with shapes (2,3) (2,)
~~~

배열 d의 모양을 바꾸면 연산이 가능해진다.
~~~python
>>> a
array([[1., 1., 1.],
       [1., 1., 1.]])
>>> d = np.arange(2).reshape(2, 1)
>>> d
array([[0],
       [1]])
>>> a+d
array([[1., 1., 1.],
       [2., 2., 2.]])
~~~

## 9. 인덱싱과 슬라이싱
인덱싱은 배열에서 인덱스로 값에 접근하는 것을 의미한다.
넘파이의 인덱싱은 일반 배열과 동일하게 인덱싱을 사용할 수 있다.
슬라이싱이란 : 키워드로 특정 범위의 원소들에 접근하는 것을 의미한다.
~~~python
>>> import numpy as np
>>> b = np.arange(12).reshape(3, 4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> bb = b[0:2, 1:3] 
>>> bb
array([[1, 2],
       [5, 6]])
~~~

## 10. 팬시 인덱싱
배열 인덱스에 다른 배열을 전달해서 원하는 요소를 선택하는 방법을 팬시 인덱싱(fancy indexing)이라고 한다.  
전달하는 배열에 숫자를 포함하고 있으면 해당 인덱스에 맞게 선택되고,  
배열에 bool값을 포함하면 True인 값을 갖는 요소만 선택된다.  
~~~python
>>> import numpy as np
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b = a > 5
>>> b
array([False, False, False, False, False, False,  True,  True,  True,
        True])
>>>
>>> a[b]
array([6, 7, 8, 9])
>>> a[a>5]
array([6, 7, 8, 9])
>>> a[a>5] =1
>>> a
array([0, 1, 2, 3, 4, 5, 1, 1, 1, 1])
~~~

## 11. 병합과 분리
 - numpy.hstack(arrays): arrays 배열을 수평으로 병합
 - numpy.vstack(arrays): arrays 배열을 수직으로 병합
 - numpy.concatenate(arrays, axis=0): arrays배열을 지정한 축 기준으로 병합
 - numpy.stack(arrays, axis=0): arrays배열을 새로운 축으로 병합
   - arrays: 병합 대상 배열(튜플)
   - axis: 작업할 대상 축 번호

~~~python
>>> import numpy as np
>>> a = np.arange(4).reshape(2,2)
>>> a
array([[0, 1],
       [2, 3]])
>>> b = np.arange(10, 14).reshape(2,2)
>>> b
array([[10, 11],
       [12, 13]])
>>> np.vstack((a, b))
array([[ 0,  1],
       [ 2,  3],
       [10, 11],
       [12, 13]])
>>> np.hstack((a, b)) 
array([[ 0,  1, 10, 11],
       [ 2,  3, 12, 13]])

>>> np.concatenate((a, b), 0) 
array([[ 0,  1],
       [ 2,  3],
       [10, 11],
       [12, 13]])
>>> np.concatenate((a, b), 1) 
array([[ 0,  1, 10, 11],
       [ 2,  3, 12, 13]])
~~~
2행 2열인 배열 a와 b를 numpy.vstack()으로 수직 병합해서 4행 2열 배열로 만들고,  
numpy.hstack()으로 수평 병합해서 2행 4열 배열로 만든다.  
numpy.vstack()함수와 유사하게 numpy.concatenate()함수에 축 번호로 0을 지정해서도 같은 결과를 얻을 수 있다.  
축 방향으로 수평, 수직 병합을 결정할 수 있으므로...  

stack()함수는 차원(축)이 새로 늘어나는 방법으로 병합을 한다.
축 번호를 지정하지 않으면 0번을 의미하고, -1은 마지막 축 번호를 의미한다.  

배열을 분리할 때 사용하는 함수는 아래와 같다.  
 - numpy.vsplit(array, indice): array 배열을 수평으로 분리
 - numpy.hsplit(array, indice): array 배열을 수직으로 분리
 - numpy.split(array, indice, axis=0): array 배열을 axis축으로 분리
   - array: 분리할 배열
   - indice: 분리할 개수 또는 인덱스
   - axis: 기준 축 번호

indice는 어떻게 나눌지를 정하는 인자인데, 정수 또는 1차원 배열을 사용할 수 있다.  
정수를 전달하면 배열을 그 수로 나누고,  
1차원 배열을 전달하면 나누고자 하는 인덱스로 사용한다.  
~~~python
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> a = np.arange(12)
>>> a
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> np.hsplit(a, 3) 
[array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
>>> np.hsplit(a, [3, 6])   
[array([0, 1, 2]), array([3, 4, 5]), array([ 6,  7,  8,  9, 10, 11])]
>>> np.hsplit(a, [3, 6, 9]) 
[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([ 9, 10, 11])]
~~~
np.hsplit(a, 3)은 indice항목에 3이 전달되었으므로 배열을 3개로 쪼개어  
각 배열은 4개 요소씩 갖는다.  
  
np.hsplit(a, [3, 6])은 3과 6을 배열로 표시했기 때문에 인덱스로 사용한다.  
이것은 [0:3], [3:6], [6:]과 같은 의미이다.  
인덱스를 좀 더 자세히 전달하려면 np.hsplit(a, [3, 6, 9])와 같이 전달할 수도 있다.  

## 12. 검색
 - ret = numpy.where(condition, [, t, f]): 조건에 맞는 요소를 찾기
   - ret: 검색 조건에 맞는 요소의 인덱스 또는 변경된 값으로 채워진 배열(튜플)
   - condition: 검색에 사용할 조건식
   - t, f: 조건에 따라 지정할 값 또는 배열, 배열의 경우 조건에 사용한 배열과 같은 shape
     - t: 조건에 맞는 값에 지정할 값이나 배열
     - f: 조건에 틀린 값에 지정할 값이나 배열
   - numpy.nonzero(array): array에서 요소 중에 0이 아닌 요소의 인덱스들을 반환(튜플)
   - numpy.all(array, [, axis]): array의 모든 요소가 True인지 검색
     - array: 검색 대상 배열
     - axis: 검색할 기준 축, 생략하면 모든 요소 검색, 지정하면 축 개수별로 결과를 반환
   - numpy.any(array, [, axis]): array의 어느 요소이든 True가 있는지 검색
~~~python
>>> import numpy as np
>>> a = np.arange(10, 20)
>>> a
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
>>> np.where(a > 15) 
(array([6, 7, 8, 9], dtype=int64),)
>>> np.where(a > 15, 1, 0)
array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
>>> a
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
~~~

만약 조건에 맞는 요소만 특정한 값으로 변경하고  
맞지 않는 요소는 기존 값을 그대로 갖게 하려면 다음과 같이 사용할 수 있다.  
~~~python
>>> a 
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
>>> np.where(a > 15, 99, a)
array([10, 11, 12, 13, 14, 15, 99, 99, 99, 99])
>>> np.where(a > 15, a, 0)
array([ 0,  0,  0,  0,  0,  0, 16, 17, 18, 19])
>>> a
array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
~~~

아래 코드는 3행 4열의 배열에서 6보다 큰 수만 검색하는 코드인데,
검색 결과는 행 번호(axis=0)만 갖는 배열과
열 번호(axis=1)만 갖는 배열 2개를 반환한다.
~~~python
>>> b = np.arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> coords = np.where(b>6)
>>> coords
(array([1, 2, 2, 2, 2], dtype=int64), array([3, 0, 1, 2, 3], dtype=int64))
>>>
~~~
여기서, 따로 떨어진 2개의 배열을 (x, y)모양의 좌표로 얻으려면  
앞서 살펴본 stack()함수를 이용해서 병합하면 된다.  
-1은 1로 바꾸어도 같다.
~~~python
>>> np.stack((coords[0], coords[1]), -1)
array([[1, 3],
       [2, 0],
       [2, 1],
       [2, 2],
       [2, 3]], dtype=int64)
~~~

배열 오소 중에 0이 아닌 요소를 찾을 때는 numpy.nonzero()함수를 사용할 수 있다.  
이 함수는 0이 아닌 요소의 인덱스를 배열로 만들어서 반환한다.  
~~~python
>>> z = np.array([0, 1, 2, 0, 1, 2])
>>> np.nonzero(z)
(array([1, 2, 4, 5], dtype=int64),)
>>> zz = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]) 
>>> zz 
array([[0, 1, 2],
       [1, 2, 0],
       [2, 0, 1]])
>>> coords = np.nonzero(zz)
>>> coords
(array([0, 0, 1, 1, 2, 2], dtype=int64), array([1, 2, 0, 1, 0, 2], dtype=int64))
>>> np.stack((coords[0], coords[1]), -1)
array([[0, 1],
       [0, 2],
       [1, 0],
       [1, 1],
       [2, 0],
       [2, 2]], dtype=int64)
~~~
numpy.nonezero()함수는 True나 False 같은 bool 값에 대해서는  
False를 0으로 간주하고 동작하므로  
numpy.where()함수처럼 조건을 만족하는 요소의 인덱스를 찾는데 사용할 수도 있다.  
  
~~~python
>>> np.nonzero(b > 6) 
(array([1, 2, 2, 2, 2], dtype=int64), array([3, 0, 1, 2, 3], dtype=int64))
>>> np.where(b > 6)
(array([1, 2, 2, 2, 2], dtype=int64), array([3, 0, 1, 2, 3], dtype=int64))
~~~

NumPy 배열의 모든 요소가 참 또는 거짓인지 확인할 때는 all()함수를 사용할 수 있다.  
~~~python
>>> t = np.array([True, True, True])
>>> np.all(t)
True
>>> t[1] = False
>>> t
array([ True, False,  True])
>>> np.all(t)
False
~~~

all함수에 축(axis)인자를 지정하지 않으면 모든 요소에 대해서 True를 만족하는지 검색하지만,  
축 인자를 지정하면 해당 축을 기준으로 True를 만족하는지 반환한다.
~~~python
>>> tt = np.array([[True, True], [False, True], [True, True]])
>>> tt
array([[ True,  True],
       [False,  True],
       [ True,  True]])
>>> np.all(tt, 0)
array([False,  True])
>>>
>>> np.all(tt, 1)   
array([ True, False,  True])
~~~
axis가 0이면 열 단위로 검사한다.  
0열은 False인 원소가 있으므로 False, 1열은 모두 True이므로, (False, True)가 리턴된다.  
axis가 1이면 행 단위로 검사한다.  
0행은 모두 True, 1행은 False인 원소가 있으므로 False, 2열은 모두 True이므로(True, False, True)가 리턴된다.  
  
NumPy 배열에 조건연산을 해서 True, False를 갖는 배열이 생성되는 것을 8절에서 봤는데,  
numpy.all()과 numpy.where()함수를 이용할 수도 있다.  
2개의 배열이 서로 같은지 다른지, 다르다면 어느 항목이 다른지를 찾을 수 있다.   
~~~python
>>> a = np.arange(10)
>>> b = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> a == b
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True])
>>> np.all(a==b) 
True
>>> b[5] = -1
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b
array([ 0,  1,  2,  3,  4, -1,  6,  7,  8,  9])
>>> np.all(a==b)
False
>>> np.where(a==b) 
(array([0, 1, 2, 3, 4, 6, 7, 8, 9], dtype=int64),)
>>> np.where(a!=b)
(array([5], dtype=int64),)
~~~
코드를 보면, 배열 a와 b는 동일했었지만, b[5] = -1연산으로 다르게 바꾸었다.  
np.all(a==b)로 두 배열이 서로 같은 값으로 채워졌는지 아닌지를 확인할 수 있고
np.where(a!=b)로 다른 요소의 인덱스를 찾을 수 있다.

이미지 작업에서는 이전 프레임과 다음 프레임 간의 픽셀 값의 변화가 있는지,  
변화가 있는 픽셀의 위치가 어디인지를 찾는 방법으로  
움직임을 감지하거나 객체 추적과 같은 작업을 하는데 이 함수들을 사용한다.  

## 13. 기초 통계 함수
 - numpy.sum(array, [, axis]): 배열의 합계 계산
 - numpy.mean(array, [, axis]): 배열의 평균 계산
 - numpy.amin(array, [, axis]): 배열의 최소 값 계산
 - numpy.min(array, [, axis]): numpy.amin()과 동일
 - numpy.amax(array, [, axis]): 배열의 최대 값 계산
 - numpy.max(array, [, axis]): numpy.amax()와 동일
   - array: 계산의 대상 배열
   - axis: 계산 기준 축, 생략하면 모든 요소를 대상

~~~python
>>> a = np.arange(12).reshape(3, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> np.sum(a)
66
# 각 열의 합을 계산한다.
>>> np.sum(a, 0)
array([12, 15, 18, 21])
# 각 행의 합을 계산한다.
>>> np.sum(a, 1)
array([ 6, 22, 38])
~~~

~~~python
# 배열 a의 평균을 구한다.
>>> np.mean(a)
5.5
# 배열 a의 각 열의 평균을 구한다.
>>> np.mean(a, 0)
array([4., 5., 6., 7.])
# 배열 a의 각 행의 평균을 구한다.
>>> np.mean(a, 1)
array([1.5, 5.5, 9.5])
# 배열 a의 최소 값을 구한다.
>>> np.amin(a)
0
# 배열 a의 각 열의 최소 값을 구한다.
>>> np.amin(a, 0)
array([0, 1, 2, 3])
# 배열 a의 각 행의 최소 값을 구한다.
>>> np.amin(a, 1)
array([0, 4, 8])
# 배열 a의 최대 값을 구한다.
>>> np.amax(a)
11
# 배열 a의 각 열의 최대 값을 구한다.
>>> np.amax(a, 0)
array([ 8,  9, 10, 11])
# 배열 a의 각 행의 최대 값을 구한다.
>>> np.amax(a, 1)
array([ 3,  7, 11])
~~~

np.min과 np.amin, np.max와 np.amax는 모두 같다.
~~~python
>>> np.amin is np.min
True
>>> np.max is np.amax
True
~~~