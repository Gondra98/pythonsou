# 계층적 군집분석 : 데이터를 단계적으로 묶어 군집을 형성하는 알고리즘
# 거리가 가까운 데이터부터 계속 묶어가는 방식
# 군집 수를 미리 정하지 않아도 됨. 구조는 덴드로그램으로 확인
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

iris = load_iris()
x = iris.data
y = iris.target
labels = iris.target_names  # ['setosa', 'versicolor', 'virginica']

df = pd.DataFrame(x, columns=iris.feature_names)
print(df.head(3))

# 스케일링 - 권장
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 계층적 군집
z = linkage(x_scaled, method='ward')

# 덴드로그램
plt.figure(figsize=(12, 5))
dendrogram(z)
plt.title('아이리스로 계층적 군집')
plt.xlabel('샘플')
plt.ylabel('거리(유클리드)')
plt.show()

# 덴드로그램을 잘라서 최대 3개의 군집 만들기
clusters = fcluster(Z=z, t=3, criterion='maxclust')

df['cluster'] = clusters
print(df.head(3))
print(df.tail(3))

# 2개 feature 시각화 (산점도)
plt.figure(figsize=(6, 5))
sns.scatterplot(x=x_scaled[:, 0], y=x_scaled[:, 1], hue=clusters, palette='Set1')
# hue=clusters : 군집결과에 따라 색을 달리 표시, palette='Set1' : 색상 스타일 지정
plt.title('군집결과')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.show()  # 꽤 비슷하게 나눠짐. 군집은 정답 라벨이 없음

print('실제 라벨 : ', y[:10])           # [0 0 0 0 0 0 0 0 0 0]
print('군집 결과 : ', clusters[:10])    # [1 1 1 1 1 1 1 1 1 1] 실제0이 군집1로 군집화됨

print('\n군집 결과 검증 ---')
print('교차표 - 실제 라벨 vs 군집결과')
ct = pd.crosstab(y, clusters)
print(ct)
# col_0   1   2   3
# row_0
# 0      49   1   0
# 1       0  27  23
# 2       0   2  48
# row_0(실제 라벨): 0-setisa, 1-versicolor, 2-virginica
# col_0(군집 결과): 1-cluster 1, 2-clusetr 2, 3-cluster 3
# setosa는 완벽히 분리되었고, versicolor와 virginica는 일부 섞인 결과 보임

print('각 실제 클래스가 가장 많이 속한 군집')
for i in range(ct.shape[0]):
    max_cluster = ct.iloc[i].idxmax()
    print(f'실제 클래스 {i} -> 군집 {max_cluster} (갯수:{ct.iloc[i].max()})')
# 실제 클래스 0 -> 군집 1 (갯수:49)
# 실제 클래스 1 -> 군집 2 (갯수:27)
# 실제 클래스 2 -> 군집 3 (갯수:48)

# 정량적 평가 : 군집 결과가 실제 정답과 얼마나 유사한지를 수치로 표현
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
# adjusted_mutual_info_score : ARI(Adjusted Rand Index) - 같은 그룹끼리 잘묶였는지 평가
ari = adjusted_mutual_info_score(y, clusters)
print(f'평가지표 : ARI - {ari:.4f}')   # 0.6713
# 해석기준 : 0.7이상(매우 잘된 그룹), 0.5~0.7(잘된 그룹), 0이하(문제있음)

# normalized_mutual_info_score : NMI(Normalized Mutual Info) - 정보량 기준 얼마나 유사한지 확인(그룹간 얼마나 같은 정보를 공유하는가)
nmi = normalized_mutual_info_score(y, clusters)
print(f'평가지표 : NMI - {nmi:.4f}')   # 0.6755  0~1 사이이며   1:완벽, 0:완전다름










