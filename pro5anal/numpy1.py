# numpy의 ndarray는 단순한 배열이라기 보다 벡터/행렬 연산도 가능한 다차원 수치 데이터 구조.
import numpy as np

ss = ['tom','james','oscar', 1, True] # 여러 타입의 자료로 구성
print(ss, ' ', type(ss))

ss2 = np.array(ss)                 # 같은 타입의 자료로만 구성 
print(print(ss2, ' ', type(ss2)))  # ['tom' 'james' 'oscar' '1' 'True']   <class 'numpy.ndarray'> 

li = list(range(1,10))
print(li)
print(li[0], ' ',id(li[0]))
print(li * 10)
print("---" * 20)
for i in li:
    print(i * 10, end=" ")

num_arr = np.array(li)
print(num_arr[0], " ", num_arr[1], " ", id(num_arr[0]), " ", id(num_arr[1]))
print(num_arr)
print(num_arr * 10)  # 벡터화 연산이 가능

print()
a = np.array([1,2,3.5,], dtype='float32')
print(a, type(a))  # ndarray는 동일 타입만 취급
# 여러 타입의 자료가 입력되면 상위 타입으로 자동 변환. int -> float -> complex -> str

print()
b = np.array([[1,2,3],[4,5,6]])
print(b, b.shape, " ", b[0, 0], ' ', b[[0]])

print()
c = np.zeros((2,2))
print(c)

d = np.ones((2,2))
print(d)

e = np.eye(3) # 대각행렬
print(e)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

print()
print(np.random.rand(5))   # 균등 분포
print(np.random.randn(5))  # 정규 분포

np.random.seed(0)    # 난수 고정
print(np.random.randn(2,3))

print(list(range(0,10)))
print(np.arange(10))

print()
# 인덱싱/슬라이싱
a = np.array([1,2,3,4,5])
print(a, ' ', a[1])   # 인덱싱
print(a[1:4])         # 슬라이싱
print(a[1:])
print(a[1:5:2])  # [시작값:끝값:증가치]
print(a[-2:])    # [4 5]

b = a              # 주소 치환
print(a[0],' ',b[0])
b[0] = 88
print(a[0],' ',b[0])

c = np.copy(a)     # 복사본 생성
print(a[0],' ',c[0])   # 88   88
b[0] = 33
print(a[0],' ',c[0])  # 33   88


