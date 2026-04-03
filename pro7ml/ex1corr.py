# 공분산 & 상관계수
# 변수 1개, 분산 = 거리
# 변수 2개, 분산 = 거리와 방향

# 공분산 행렬 (2차원)
# Cov(X, Y) = [ Var(X)   Cov(X,Y) ]
#             [ Cov(Y,X) Var(Y)  ]
# 대각선: 각 변수의 분산 (Var)
# 비대각선: 두 변수 간 공분산 (Cov)
#       공분산 > 0 이면 우상향
#       공분산 < 0 이면 우하향

import numpy as np

print(np.cov(np.arange(1,6), np.arange(2,7)))
# 우상향
# 2.5 2.5
# 2.5 2.5

print(np.cov(np.arange(1,6), np.array([3,3,3,3,3])))
# 직선
# 2.5 0.
# 0.  0.

print(np.cov(np.arange(1,6), np.arange(6,1,-1)))
# 우하향
#  2.5 -2.5
# -2.5  2.5

x=[8,3,6,6,9,4,3,9,3,4]
y=[6,2,4,6,9,5,1,8,4,5]
print(np.mean(x), np.var(x))
print(np.mean(y), np.var(y))

import matplotlib.pyplot as plt
# x,y의 관계성을 시각화 하여 확인
# plt.plot(x,y,'o')
# plt.show()

print('x,y의 공분산 : ', np.cov(x,y)[0,1])

x2=np.array(x)*10
y2=np.array(y)
print('x2,y2의 공분산 : ', np.cov(x2,y2)[0,1])

# plt.plot(x2,y2,'o')
# plt.show()

# 두 데이터의 단위에 따라 패턴이 일치할지라도
# 공분산의 크기가 달라지므로 절대적 크기 판단이 어렵다
# 따라서 공분산을 표준화하여 [-1,1] 범위로 만든 것이 상관계수(r)

print('x,y의 상관행렬 : ', np.corrcoef(x,y))        # 피어슨 상관 행렬
# - 대각선: 항상 1 (자기 자신과의 상관)
# - 비대각선: X와 Y 사이 상관계수

print('x,y의 상관계수 : ', np.corrcoef(x,y)[0,1])  # 피어슨 상관 계수
# ML에서 상관관계 해석 기준 (경험적)
#  r > 0.3   : 양적(positive) 상관 관계
#  r < -0.3  : 음적(negative) 상관 관계
#  -0.3 <= r <= 0.3 : 의미 없는 상관 관계

# 비선형 데이터 y=x^2
m=np.array([-3,-2,-1,0,1,2,3])
n=np.array([9,4,1,0,1,4,9])

print(np.cov(m,n)[0,1])         # 공분산 0
print(np.corrcoef(m,n)[0,1])    # 상관계수 0
plt.plot(m,n,'o')
plt.show()

# 비선형 데이터의 경우 공분산과 상관계수는 의미가 없다.
# 반드시 선형 데이터를 사용해야 의미가 있다.