# 배열의 행, 열 추가 ...

import numpy as np

aa = np.eye(3)  # 주대각이 1인 단일 행렬 만듦
print(aa)

bb = np.c_[aa, aa[2]]  # 2열과 동일한 열 추가
print(bb)

cc = np.r_[aa, [aa[2]]]  # 2행과 동일한 행 추가
print(cc)

print('--append, insert, delete ---')
a = np.array([1,2,3])
print(a)
# b = np.append(a, [4,5])
b = np.append(a, [4,5], axis=0) # 행 기준
print(b)

c = np.insert(a, 0, [6,7])
print(c)

d = np.delete(a, 1)
print(d)

print()
aa = np.arange(1,10).reshape(3,3)
print(aa)
print(np.insert(aa, 1, 99))          # axis 지정 없으면 1차원으로 병합(flatten) 후 삽입
print(np.insert(aa, 1, 99, axis=0))  # axis=0: 행(Row) 방향. index 1 위치에 새로운 행 삽입
print(np.insert(aa, 1, 99, axis=1))  # axis=1: 열(Column) 방향. index 1 위치에 새로운 열 삽입

# 요약: axis=0은 위에서 아래로(행 추가/연산), axis=1은 왼쪽에서 오른쪽으로(열 추가/연산)

print()
# 조건 연산 where(조건, 참, 거짓)
x = np.array([1,2,3])
y = np.array([4,5,6])
conditionData = np.array([True, False, True])
result = np.where(conditionData, x, y)
print(result)   # [1 5 3]

print()
aa = np.where(x >= 2)
print(aa)    # (array([1, 2]),) 인덱스 출력
print(x[aa])

print()
# 배열 결합
kbs = np.concatenate([x,y])
print(kbs)

# 배열 분할
mbc, sbs = np.split(kbs,2)
print(mbc)
print(sbs)

print()
a = np.arange(1, 17).reshape(4,4)
print(a)
# 배열 분할 (hsplit: 좌우/열 방향, vsplit: 상하/행 방향)
print("---" * 10)
x1, x2 = np.hsplit(a,2)  # 수평(Horizontal) 분할: 열을 기준으로 나눔
print(x1)
print(x2)
print()
print(np.vsplit(a,2))    # 수직(Vertical) 분할: 행을 기준으로 나눔

print('\n표본 추출(sampling) - 복원, 비복원')
li = np.array([1,2,3,4,5,6,7])

# 복원
for _ in range(5):
    print(li[np.random.randint(0, len(li) - 1)], end= ' ')  # 중복 허용

print()
# 비복원
import random
print(random.sample(li.tolist(), 5))  # random.sample()은 대상이 list type  # 중복 불가

print()
# choice(대상, 개수, replace=True/False) : 샘플링 함수
print(np.random.choice(range(1,46),6))                # 1~45 사이의 숫자 중 6개 추출
print(np.random.choice(range(1,46),6, replace=True))  # replace=True : 복원 추출(중복 허용, 기본값)
print(np.random.choice(range(1,46),6, replace=False)) # replace=False : 비복원 추출(중복 불가)
