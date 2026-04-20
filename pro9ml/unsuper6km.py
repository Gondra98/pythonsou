# 쇼핑몰 고객 세분화 연습

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.cluster import KMeans

# 가상의 고객 데이터 생성
np.random.seed(0)
n_customers = 200   # 고객은 200명
annual_spending = np.random.normal(50000, 15000, n_customers)    # 연간 지출액
monthly_visits = np.random.normal(5, 2, n_customers)    # 월 방문 횟수

# 구간 나누기(음수 제거 - clip)
# a = np.array([-3.2, -0.5, 1.7]) -> np.clip(a, 0, 1) -> [0, 0, 1]
annual_spending = np.clip(annual_spending, 0, None)
monthly_visits = np.clip(monthly_visits, 0, None)

data = pd.DataFrame({
    'annual spending': annual_spending,
    'monthly visits': monthly_visits
})
print(data.head(), data.shape)
#    annual spending  monthly visits
# 0     76460.785190        4.261636
# 1     56002.358126        4.521242
# 2     64681.069762        7.199319
# 3     83613.397988        6.310527
# 4     78013.369852        6.280263 (200, 2)

# 산포도
plt.scatter(data['annual spending'], data['monthly visits'])
plt.xlabel('연간 지출액')
plt.ylabel('달별 방문수')
plt.title('소비자 분포')
plt.show()

# KMeans 군집화
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(data)

# cluster 별로 데이터 보기
data['cluster'] = clusters
# print(data.head(3))

# 군집 결과 시각화
for cluster_id in np.unique(clusters):
    clusters_data = data[data['cluster'] == cluster_id]
    plt.scatter(clusters_data['annual spending'], 
                clusters_data['monthly visits'],
                label=f'군집{cluster_id}')
# 중심점
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            c='black', marker='X', s=200, label='중심점')
plt.xlabel('연간 지출액')
plt.ylabel('달별 방문수')
plt.title('소비자 군집 현황')
plt.legend()
plt.show()







