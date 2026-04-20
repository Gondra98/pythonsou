# KMeans : iris dataset - 군집분석, 정량평가, 클러스터별 평균비교(ANOVA)

import os
os.environ['OMP_NUM_THREADS'] = '1' # 경고 무시

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
# adjusted_rand_score : 군집 vs 실제 라벨 비교
# normalized_mutual_info_score : 정보량 기반 유사도(같은 정보 공유)
# silhouette_score : 군집 자체 품질 평가(군집에 잘 속해 있는가 확인)
from sklearn.decomposition import PCA   # 4차원 -> 2차원으로 압축


iris = load_iris()
x = iris.data
y = iris.target
features_names = iris.feature_names

df = pd.DataFrame(x, columns=features_names)
print('iris data shape : ', x.shape)    # (150, 4)

# 스케일링
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x_scaled[:2])
# [[-0.90068117  1.01900435 -1.34022653 -1.3154443 ]
#  [-1.14301691 -0.13197948 -1.34022653 -1.3154443 ]]

# PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print('pca 설명 분산 비율 : ', pca.explained_variance_ratio_)   # [0.72962445 0.22850762]
print(x_pca[:2])
# [[-2.26470281  0.4800266 ]
#  [-2.08096115 -0.67413356]]

# KMeans 모델
k = 3
kmeans = KMeans(
    n_clusters=k, init='k-means++', n_init=10, random_state=42
    # n_init=10 10회 수행시 가장 오차(inertia)가 작은 결과를 선택
)

clusters = kmeans.fit_predict(x_scaled)
df['cluster'] = clusters
print('클러스터 중심 값 : ', kmeans.cluster_centers_)
# [[-0.05021989 -0.88337647  0.34773781  0.2815273 ]
#  [-1.01457897  0.85326268 -1.30498732 -1.25489349]
#  [ 1.13597027  0.08842168  0.99615451  1.01752612]]

# 시각화(PCA기반. 왜? 4개의 열을 2차원차트에 표현불가. 그래서 두개로 표현)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
print('pca 설명 분산 비율 : ', pca.explained_variance_ratio_)   # [0.72962445 0.22850762]
print(x_pca[:2])

plt.figure(figsize=(6, 5))
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=clusters, palette='Set1')
plt.title('KMeans Clustering')
plt.xlabel('PC1(제1주성분)')
plt.ylabel('PC2(제2주성분)')
plt.show()

# 실제 라벨과 군집 비교(교차표)
ct = pd.crosstab(y, clusters)
print(ct)
# col_0   0   1   2     <- 열:군집번호(KMeans결과)
# row_0
# 0       0  50   0     <-- setosa
# 1      39   0  11     <-- versicolor (?)
# 2      14   0  36     <-- verginica (?)
# 행:실제라벨(iris)

print('클래스별 대표 군집')
for i in range(ct.shape[0]):
    max_cluster = ct.iloc[i].idxmax()
    print(f'실제 클래스 {i} -> 군집 {max_cluster}')
# 실제 클래스 0 -> 군집 1
# 실제 클래스 1 -> 군집 0
# 실제 클래스 2 -> 군집 2

print('정량 평가 ---')
ari = adjusted_rand_score(y, clusters)
nmi = normalized_mutual_info_score(y, clusters)
sil_score = silhouette_score(x_scaled, clusters)
print(f'ARI : {ari:.4f}')
print(f'NMI : {nmi:.4f}')
print(f'Silhouette_Score : {sil_score:.4f}')
# ARI : 0.6201
# NMI : 0.6595
# Silhouette_Score : 0.4599     <- 1에 근사할수록 좋음. 0또는 음수면 잘못된 군집
# 좋은 군집이란 군집 내 요소끼리는 가깝고, 다른 군집 간에는 거리가 멀다.

# k=3을 사용했는데 과연 3이 합리적인지 확인 : 엘보우
initia_list = []
k_range = range(1, 10)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(x_scaled)
    initia_list.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, initia_list, marker='o')
plt.title('엘보우 기법')
plt.xlabel('클러스터 수(k)')
plt.ylabel('initia')
plt.show()  # k가 3인 경우가 가장 적당

# 실제 vs 군집 비교 시각화
plt.figure(figsize=(12, 5))
# 실제 라벨
plt.subplot(1, 2, 1)
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y, palette='Set1')
plt.title('실제 라벨')

# 군집 결과
plt.subplot(1, 2, 2)
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=clusters + 1, palette='Set1')
plt.title('군집 결과')
plt.show()

# 클러스터별 평균 분석
cluster_mean = df.groupby('cluster').mean()
print('클러스터별 평균 : ', cluster_mean)
# 클러스터별 평균 :           sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# cluster
# 0                 5.801887          2.673585           4.369811          1.413208
# 1                 5.006000          3.428000           1.462000          0.246000
# 2                 6.780851          3.095745           5.510638          1.972340

print('------------------------------------')
# 군집 3개 : 군집 간 평균차이 검정(ANOVA)
# 귀무: 군집 간 평균에 차이가 없다.
# 연구: 군집 간 평균에 차이가 있다.
from scipy.stats import f_oneway

for col in features_names:  # 각 군집별 데이터 분리
    group0 = df[df['cluster'] == 0][col]
    group1 = df[df['cluster'] == 1][col]
    group2 = df[df['cluster'] == 2][col]
    # ANOVA 수행
    f_stat,p_val = f_oneway(group0, group1, group2)
    print(f'{col} : f-statistic:{f_stat}, p-value:{p_val:.4f}')
    
    # 해석
    if p_val >= 0.05:
        print('군집간 평균에 차이가 없다(유의하지 않다)')
    else:
        print('군집간 평균에 차이가 있다(유의하다. 우연이 아니다)')
# sepal length (cm) : f-statistic:218.5710027061961, p-value:0.0000
# 군집간 평균에 차이가 있다(유의하다. 우연이 아니다)
# sepal width (cm) : f-statistic:79.90002498360766, p-value:0.0000
# 군집간 평균에 차이가 있다(유의하다. 우연이 아니다)
# petal length (cm) : f-statistic:860.6366870013799, p-value:0.0000
# 군집간 평균에 차이가 있다(유의하다. 우연이 아니다)
# petal width (cm) : f-statistic:525.6987697249469, p-value:0.0000
# 군집간 평균에 차이가 있다(유의하다. 우연이 아니다)

# KMeans가 꽃받침,꽃잎 길이/너비를 제대로 군집분석했음을 알 수 있다.


# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# petal length로 작업
feature = 'petal length (cm)'
tukey = pairwise_tukeyhsd(
    endog=df[feature], groups=df['cluster'], alpha=0.05
)
print('tukeyhsd 결과(petal length) : ', tukey)
# tukeyhsd 결과(petal length) :  Multiple Comparison of Means - Tukey HSD, FWER=0.05
# ===================================================
# group1 group2 meandiff p-adj  lower   upper  reject
# ---------------------------------------------------
#      0      1  -2.9078   0.0 -3.1405 -2.6751   True
#      0      2   1.1408   0.0  0.9043  1.3773   True
#      1      2   4.0486   0.0  3.8088  4.2884   True
# ---------------------------------------------------

# 사후 검정 시각화
tukey.plot_simultaneous(figsize=(6, 4))
plt.title(f'tukeyhsd - {feature}')
plt.xlabel('평균 차이')
plt.show()

print()
# 군집별 Boxplot
for col in features_names:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x='cluster', y=col, data=df)
    plt.title(f'{col} by cluster')
    plt.show()

print() # 클러스터 평균분석 마지막 열에 Type 추가
cluster_mean['label'] = ['Type A', 'Type B', 'Type C']
print(cluster_mean)
#          sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)   label
# cluster
# 0                 5.801887          2.673585           4.369811          1.413208  Type A
# 1                 5.006000          3.428000           1.462000          0.246000  Type B
# 2                 6.780851          3.095745           5.510638          1.972340  Type C
