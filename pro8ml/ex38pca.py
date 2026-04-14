# PCA(주성분분석) : 선형대수 관점에서, 입력데이터의 공분산 행렬을 고윳값 분해하고
# 이렇게 구한 고유벡터에 입력 데이터를 선형변환하는 것이다.
# 이 고유벡터가 PCA의 주성분 벡터로서 입력 데이터의 분산이 큰 방향을 나타낸다.
# 입력 데이터의 성질을 최대한 유지한 상태로 고차원을 저차원 데이터로 변환하는 기법

# iris data로 차원 축소
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
n = 10
x = iris.data[:n, :2]   # sepal width, height 열만 선택
print('차원 축소 전 : x: ', x, x.shape, type(x))
print(x.T)
# 차원 축소 전 : x:  [[5.1 3.5]
#  [4.9 3. ]
#  [4.7 3.2]
#  [4.6 3.1]
#  [5.  3.6]
#  [5.4 3.9]
#  [4.6 3.4]
#  [5.  3.4]
#  [4.4 2.9]
#  [4.9 3.1]] (10, 2) <class 'numpy.ndarray'>
# [[5.1 4.9 4.7 4.6 5.  5.4 4.6 5.  4.4 4.9]
#  [3.5 3.  3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1]]

# 시각화
plt.plot(x.T, 'o:')
plt.xticks(range(2), ['꽃받침길이', '꽃받침너비'])
plt.grid(True)
plt.title('아이리스 크기 특성')
plt.xlabel('특성의 종류')
plt.ylabel('특성값')
plt.xlim(-0.5, 2)
plt.ylim(2.5, 6)
plt.legend(['표본 {}'.format(i + 1) for i in range(n)])
plt.show()