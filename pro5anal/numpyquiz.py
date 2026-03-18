import numpy as np

# 1) step1 : array 관련 문제
# 정규분포를 따르는 난수를 이용하여 5행 4열 구조의 다차원 배열 객체를 생성하고, 각 행 단위로 합계, 최댓값을 구하시오.

a = np.random.randn(5, 4)

# 최대값
row_sum = np.sum(a, axis=1)
# print(row_sum)

# 최대값
row_max = np.max(a, axis=1)
# print(row_max)


# 2) step2 : indexing 관련문제

#  문2-1) 6행 6열의 다차원 zero 행렬 객체를 생성한 후 다음과 같이 indexing 하시오.
k = np.zeros((6,6))
# print(k)

#    조건1> 36개의 셀에 1~36까지 정수 채우기
k = np.arange(1, 37).reshape(6, 6)
# print(k)

#    조건2> 2번째 행 전체 원소 출력하기 
# print(k[1])

#    조건3> 5번째 열 전체 원소 출력하기
# print(k.T[4])

#    조건4> 15~29 까지 아래 처럼 출력하기
            #   [[15.  16.  17.]

            #   [21.  22.  23]

            #   [27.  28.  29.]]
# print(k[2:5, 2:5])
