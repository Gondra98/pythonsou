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

# 시각화1
# plt.plot(x.T, 'o:')
# plt.xticks(range(2), ['꽃받침길이', '꽃받침너비'])
# plt.grid(True)
# plt.title('아이리스 크기 특성')
# plt.xlabel('특성의 종류')
# plt.ylabel('특성값')
# plt.xlim(-0.5, 2)
# plt.ylim(2.5, 6)
# plt.legend(['표본 {}'.format(i + 1) for i in range(n)])
# plt.show()

# 시각화2 : 산점도
df = pd.DataFrame(x)
print(df)
# ax = sns.scatterplot(x=df[0], y=df[1], markers='s', s=100, color='b')
ax = sns.scatterplot(x=0, y=1, data=df, marker='s', s=100, color='b')
# 각 점에 대해 text 표시
for i in range(n):
    ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, '표본{}'.format(i+1))
plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침폭')
plt.title('아이리스 특성')
plt.axis('equal')
plt.show()

# 위 두개의 그래프 결과 두 변수는 공통적인 특징이 있으므로 차원축소의 근거가 있다고 판단.
# PCA를 진행 - 선형변환을 통해 차원을 축소
# 순서1: 입력 데이터의 공분산 행렬을 생성한다.
# 순서2: 공분산 행렬의 고유벡터와 고윳값(고유벡터 크기)을 계산한다.
# 순서3: 고윳값이 큰 순서대로 k개(PCA 변환 차수 만큼) 만큼 고유벡터 추출
# 순서4: 고윳값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환한다.
# sklearn의 PCA를 이용하면 순서대로 진행을 함

pca1 = PCA(n_components=1)      # 변환할 차원수
x_low = pca1.fit_transform(x)   # 특징 행렬을 낮은 차원의 근사행렬로 변환
print('x_low : ', x_low, ' ', x_low.shape)
# 주성분 값 원복하기
x2 = pca1.inverse_transform(x_low)
print('원복 후 x2: ', x2, ' ', x2.shape)
#      0    1
# 0  5.1  3.5
# 1  4.9  3.0
# 2  4.7  3.2
# 3  4.6  3.1
# 4  5.0  3.6
# 5  5.4  3.9
# 6  4.6  3.4
# 7  5.0  3.4
# 8  4.4  2.9
# 9  4.9  3.1
# x_low :  [[ 0.30270263]
#  [-0.1990931 ]
#  [-0.18962889]
#  [-0.33097106]
#  [ 0.30743473]
#  [ 0.79976625]
#  [-0.11185966]
#  [ 0.16136046]
#  [-0.61365539]
#  [-0.12605597]]   (10, 1)
# 원복 후 x2:  [[5.06676112 3.53108532]
#  [4.7240094  3.1645881 ]
#  [4.73047393 3.17150049]
#  [4.63393012 3.06826822]
#  [5.06999338 3.53454152]
#  [5.40628057 3.89412635]
#  [4.78359423 3.22830091]
#  [4.97021731 3.42785306]
#  [4.44084251 2.86180369]
#  [4.77389743 3.21793233]]   (10, 2)
print('원본 : ', x[0, :])       # [5.1 3.5]
print('주성분 : ', x_low[0])    # [0.30270263]
print('원본 : ', x2[0, :])       # [5.06676112 3.53108532]

# 주성분 분석값을 기반으로 시각화
# pca 방향벡터
pc1 = pca1.components_[0]   # components_ : 주성분벡터
mean = x.mean(axis=0)   # 데이터 평균(중심점)

df = pd.DataFrame(x)
ax = sns.scatterplot(x=0, y=1, data=df, marker='s', s=100, color='b')
# 각 점에 대해 text 표시
for i in range(n):
    ax.text(x[i, 0] - 0.05, x[i, 1] + 0.03, f'표본{i+1}')

# PCA 축 (화살표)
plt.quiver(
    mean[0], mean[1],   # 시작점(평균)
    pc1[0], pc1[1],     # 방향
    scale=3, color='r', width=0.01
)

plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침폭')
plt.title('아이리스 특성 + 제1주성분')
plt.axis('equal')
plt.grid(True)
plt.show()


print('***' * 10)
# 원본 열 4개를 차원축소해 2개의 열로 변환 후 SVM 분류 모델을 작성
x = iris.data
print(x[0, :])  # [5.1 3.5 1.4 0.2]
pca2 = PCA(n_components=2)
x_low2 = pca2.fit_transform(x)
print('x_low2 : ', x_low2[0, :], ' ', x_low2.shape)     # [-2.68412563  0.31939725]   (150, 2)
# 변동성 비율 확인
print(pca2.explained_variance_ratio_)   # [0.92461872 0.05306648]
x4 = pca2.inverse_transform(x_low2)
print('최초 자료 : ', x[0])
print('차원축소 : ', x_low2[0])
print('차원복귀 : ', x4[0, :])
# 최초 자료 :  [5.1 3.5 1.4 0.2]
# 차원축소 :  [-2.68412563  0.31939725]
# 차원복귀 :  [5.08303897 3.51741393 1.40321372 0.21353169]
print()
iris1 = pd.DataFrame(x, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
print(iris1.head(3))
iris2 = pd.DataFrame(x_low2, columns=['var1', 'var2'])
print(iris2.head(3))

from sklearn import svm, metrics
feature1 = iris1.values
print(feature1[:3])
label = iris.target
print(label[:3])

print('원본 데이터로 SVM 분류 모델 작성')
model1 = svm.SVC(C=0.1, random_state=0).fit(feature1, label)
pred1 = model1.predict(feature1)
print('model1 accuracy : ', metrics.accuracy_score(label, pred1))   # 0.94

print('PCA 데이터로 SVM 분류 모델 작성')
feature2 = iris2.values
print(feature2[:3])
print(label[:3])

model2 = svm.SVC(C=0.1, random_state=0).fit(feature2, label)
pred2 = model2.predict(feature2)
print('model2 accuracy : ', metrics.accuracy_score(label, pred2))   # 0.9466666666666667
