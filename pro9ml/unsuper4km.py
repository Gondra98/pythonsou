# 비계층적 군집분석(Non-hierachical Clustering)은 주어진 데이터를 k개의 군집으로 나눈다. 
# 원하는 군집의 수 k는 사전에 지정 (알고 있다고 가정) 한다.
# k-평균 군집화(M-means Clustering) 알고리즘은 군집의 중심이 되는 k개의 seed(씨드) 점들을 선택하여 
# 그 seed 점(중심점)과 거리가 가까운 개체들을 그룹화하는 방법이다. 
# 알고리즘은 다음과 같다.
# 1) K개의 중심점을 임의로 배치한다.
# 2) 모든 자료와 K개의 중심점과 거리를 계산하여 가장 가까운 중심점의 군집으로 할당한다.
# 3) 군집의 중심을 구한다. (평균을 구한다.)
# 4) 정지규칙에 이를 때까지 2~3단계를 반복한다.
#   - 군집의 변화가 없을때
#   - 중심점의 이동이 임계값 이하일 때
#   - 왜곡값(distortion, 각각의 클러스터의 거리제곱의 총합) 줄어들었다가 다시 늘어나는 지점

# 실습1 - make_blobs 사용
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import koreanize_matplotlib

x, _ = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

print(x[:3], ' ', x.shape)
# [[2.60509732 1.22529553]
#  [0.5323772  3.31338909]
#  [0.802314   4.38196181]]   (150, 2)

# 산점도
plt.scatter(x[:, 0], x[:, 1], c='gray', marker='o', s=50)
plt.grid(True)
plt.show()

# KMeans 모델 작성
# cluster의 중심을 선택하는 방법
# init_centroid = 'random'        # cluster의 중심을 임의로 선택
init_centroid = 'k-means++'     # cluster의 중심을 k-means++로 선택. 중심을 최대한 멀리하는 방법
# kmodel = KMeans(n_clusters=3, init=init_centroid, n_init=10, random_state=0)
# n_init=10 : KMeans를 10회 실행 - 가장 좋은 결과(오차 최소값)를 선택
kmodel = KMeans(n_clusters=3, init=init_centroid, random_state=0)
pred = kmodel.fit_predict(x)    # 클러스터링으로 구분한 결과
print('pred : ', pred)
# pred :  [1 2 2 2 1 2 2 1 0 2 1 0 0 2 2 0 0 1 0 1 2 1 2 2 0 1 1 2 0 1 0 0 0 0 2 1 1
#  1 2 2 0 0 2 1 1 1 0 2 0 2 1 2 2 1 1 0 2 1 0 2 0 0 0 0 2 0 2 1 2 2 2 1 1 2
#  1 2 2 0 0 2 1 1 2 2 1 1 1 0 0 1 1 2 1 2 1 2 0 0 1 1 1 1 0 1 1 2 0 2 2 2 0
#  2 1 0 2 0 2 2 0 0 2 1 2 2 1 1 0 1 0 0 0 0 1 0 0 0 2 0 1 0 2 2 1 1 0 0 0 0
#  1 1]

# 각 그룹별 보기
# print(x[pred == 0])
# print(x[pred == 1])
# print(x[pred == 2])
print('중심점 : ', kmodel.cluster_centers_)
# [[-1.5947298   2.92236966]
#  [ 2.06521743  0.96137409]
#  [ 0.9329651   4.35420712]]

# 시각화
plt.scatter(x[pred == 0, 0], x[pred == 0, 1], c='red', marker='o', s=50, label='cluster1')
plt.scatter(x[pred == 1, 0], x[pred == 1, 1], c='green', marker='s', s=50, label='cluster2')
plt.scatter(x[pred == 2, 0], x[pred == 2, 1], c='blue', marker='v', s=50, label='cluster3')
plt.scatter(kmodel.cluster_centers_[:, 0], kmodel.cluster_centers_[:, 1], c='black', marker='+', s=60, label='center')    # 각 군집 중심점 표시
plt.legend()
plt.grid(True)
plt.show()

# KMeans의 k값은?   elbow or silhouette 기법을 이용해 k값 얻기
# 1) elbow
def elbow(x):
    sse = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init=init_centroid, random_state=0)
        km.fit(x)
        sse.append(km.inertia_)
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('군집수')
    plt.ylabel('SSE')
    plt.show()

elbow(x)

# 2) silhouette
'''
실루엣(silhouette) 기법
  클러스터링의 품질을 정량적으로 계산해 주는 방법이다.
  클러스터의 개수가 최적화되어 있으면 실루엣 계수의 값은 1에 가까운 값이 된다.
  실루엣 기법은 k-means 클러스터링 기법 이외에 다른 클러스터링에도 적용이 가능하다
'''
import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm

# 데이터 X와 X를 임의의 클러스터 개수로 계산한 k-means 결과인 y_km을 인자로 받아 각 클러스터에 속하는 데이터의 실루엣 계수값을 수평 막대 그래프로 그려주는 함수를 작성함.
# y_km의 고유값을 멤버로 하는 numpy 배열을 cluster_labels에 저장. y_km의 고유값 개수는 클러스터의 개수와 동일함.

def plotSilhouette(x, pred):
    cluster_labels = np.unique(pred)
    n_clusters = cluster_labels.shape[0]   # 클러스터 개수를 n_clusters에 저장
    sil_val = silhouette_samples(x, pred, metric='euclidean')  # 실루엣 계수를 계산
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        # 각 클러스터에 속하는 데이터들에 대한 실루엣 값을 수평 막대 그래프로 그려주기
        c_sil_value = sil_val[pred == c]
        c_sil_value.sort()
        y_ax_upper += len(c_sil_value)

        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_value, height=1.0, edgecolor='none')
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_sil_value)

    sil_avg = np.mean(sil_val)         # 평균 저장

    plt.axvline(sil_avg, color='red', linestyle='--')  # 계산된 실루엣 계수의 평균값을 빨간 점선으로 표시
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('클러스터')
    plt.xlabel('실루엣 개수')
    plt.show() 

'''
그래프를 보면 클러스터 1~3 에 속하는 데이터들의 실루엣 계수가 0으로 된 값이 아무것도 없으며, 실루엣 계수의 평균이 0.7 보다 크므로 잘 분류된 결과라 볼 수 있다.
'''
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
km = KMeans(n_clusters=3, random_state=0) 
y_km = km.fit_predict(X)

plotSilhouette(X, y_km)







