# 배열 연산 
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float32)   # 2차원 배열
# x = np.array([[1,2],[3,4]])  -- 같은 값
print(x, ' ', x.dtype)  
y = np.arange(5,9).reshape((2,2))      # 구조변경 reshape((2,2)) : 1차원 배열 -> 2차원 배열
y = y.astype(np.float32)
print(y, ' ', y.dtype)

print('더하기')
print(x + y)     # 파이썬 연산자 또는 함수(느림)
print(np.add(x,y))   # numpy 함수 (유니버셜 함수)(빠름)

print('빼기')
print(x - y)
print(np.subtract(x,y))

print('곱하기')
print(x * y)
print(np.multiply(x,y))

print("나누기")
print(x / y)
print(np.divide(x,y))

print('\ndot은 numpy 모듈의 함수나 배열 객체의 인스턴트 메소드로 사용이 가능')
v = np.array([9,10])
w = np.array([11,12])
print(v * w)  # 요소별 곱셈 9 * 11, 10 * 12

# 벡터의 내적(행렬곱)
print(v.dot(w))  # 내적의 결과는 스칼라(크기만 있고 방향은 없음)
print(np.dot(v,w))  # 9 * 11 + 10 * 12
print(np.dot(x,v))

print()
# 배열 계산 함수
print(x)
print(np.mean(x), ' ', np.var(x))
print(np.max(x), ' ', np.min(x))
print(np.argmax(x), ' ', np.argmin(x))  # index 반환
print(np.cumsum(x))  # 누적합
print(np.cumprod(x))  # 누적곱

print()
names1 = np.array(['tom', 'james', 'tom','oscar'])
names2 = np.array(['tom', 'page', 'john'])
print(np.unique(names1))
print(np.intersect1d(names1,names2))  # 교집합
print(np.intersect1d(names1,names2, assume_unique=True))  # 교집합  # assume_unique=True : 중복 허용
print(np.union1d(names1,names2))  # 합집합

print('\n전치(Transpose) - 2차원 배열에서 행과 열의 위치를 바꾸는 것')
print(x)
print(x.T)
print(x.transpose())
print(x.swapaxes(0,1))

print('\nBroadcasting : 크기가 다른 배열 간의 연산 - 작은 배열을 여러 번 반복해 큰 배열과 연산을 하는것')
x = np.arange(1,10).reshape(3,3)
y = np.array([1,0,1])
print(x)
print(y)
print(x + y)

np.savetxt("my.txt",x)  # 배열 flie i/o  loadtxt() : 읽기 가능



