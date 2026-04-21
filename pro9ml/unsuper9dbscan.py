# DBSCAN 이란?
# DBSCAN(Density-Based Spatial Clustering of Applications with Noise)은 머신 러닝에 
# 주로 사용되는 클러스터링 알고리즘으로 Multi Dimension의 데이터를 밀도 기반으로 서로 
# 가까운 데이터 포인트를 함께 그룹화하는 알고리즘이다.
# DBSCAN은 밀도가 다양하거나 모양이 불규칙한 클러스터가 있는 데이터와 같이 모양이 
# 잘 정의되지 않은 데이터를 처리할 때 유용하게 사용 가능하다.

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# 샘플 데이터 생성
x, y = make_moons(n_samples=200, noise=0.05, shuffle=True, random_state=0)
print(x[:5], x.shape)
# [[ 0.81680544  0.5216447 ]
#  [ 1.61859642 -0.37982927]
#  [-0.02126953  0.27372826]
#  [-1.02181041 -0.07543984]
#  [ 1.76654633 -0.17069874]] (200, 2)
print(y[:5], y.shape)
# [0 1 1 0 1] (200,)

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

print('KMeans로 군집 분류')
km = KMeans(n_clusters=2, init='k-means++', random_state=0)
pred1 = km.fit_predict(x)
print('km 예측 군집 id : ', pred1[:10]) # [1 1 0 0 1 1 1 1 1 1]

# km 결과 시각화
def plotResult(x, pr, centers=None):
    plt.scatter(x[pr==0, 0], x[pr==0, 1], c='blue', marker='o', s=40, label='cluster1')
    plt.scatter(x[pr==1, 0], x[pr==1, 1], c='red', marker='s', s=40, label='cluster2')
    if centers is not None:           # ✅ KMeans일 때만 centroid 표시
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='+', s=200, label='centroid')
    plt.title('Clustering Result')
    plt.legend()
    plt.show()


plotResult(x, pred1, km.cluster_centers_)     # 설정에 따라 무조건 2개로 분류, 반달을 기준으로 자르기됨

print('DBSCAN으로 군집 분류')
db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')  
# eps=0.2 - 샘플간 최대거리, min_samples - 점에 대한 이웃 샘플수(반경 포인트들의 최소 갯수)
pred2 = db.fit_predict(x)
print('DBSCAN 군집 id : ', pred2[:10])  # [0 1 1 0 1 1 0 1 0 1]
print('군집 종류 : ', set(pred2))       # {np.int64(0), np.int64(1)}   0,1 이상치는 없는 상태

plotResult(x, pred2)    # 분류가 제대로 됨. 모양따라 군집형성
# KMeans는 k개에 군집의 갯수를 맞추고, DBSCAN은 밀도에 의해 형태를 맞춘다.




