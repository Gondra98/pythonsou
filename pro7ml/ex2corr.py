# 공분산 & 상관계수
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv"
data = pd.read_csv(uri)
data.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 264 entries, 0 to 263
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   친밀도     264 non-null    int64
#  1   적절성     264 non-null    int64
#  2   만족도     264 non-null    int64
# dtypes: int64(3)
# memory usage: 6.3 KB

# print(sorted(data.친밀도.unique()))     # [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5)]
# print(sorted(data.적절성.unique()))     # [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5)]
# print(sorted(data.만족도.unique()))     # [np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5)]

print("공분산")
print(data.cov())
#        친밀도      적절성      만족도
# 친밀도  0.941569  0.416422  0.375663
# 적절성  0.416422  0.739011  0.546333
# 만족도  0.375663  0.546333  0.686816

print("상관계수")
print(data.corr())
#           친밀도    적절성     만족도
# 친밀도  1.000000  0.499209  0.467145
# 적절성  0.499209  1.000000  0.766853
# 만족도  0.467145  0.766853  1.000000
print(data.corr(method='pearson'))      # 연속형 변수, 정규성 -> 모수검정
# print(data.corr(method='spearman'))     # 서열척도, 비정규성  -> 비모수검정
# print(data.corr(method='kendal'))       # 서열척도, 비정규성

# 만족도에 따른 다른 특성과의 관계
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))

# 시각화
data.plot(kind='scatter', x='만족도', y='적절성')
plt.show()

from pandas.plotting import scatter_matrix
attr = ['친밀도','적절성','만족도']
scatter_matrix(data[attr], figsize=(10,6))
plt.show()

import seaborn as sns
sns.heatmap(data.corr(), annot=True)
plt.show()

# heatmap에 텍스트 표시 추가사항 적용해 보기
corr = data.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)  # 상관계수값 표시
mask[np.triu_indices_from(mask)] = True
# Draw the heatmap with the mask and correct aspect ratio
vmax = np.abs(corr.values[~mask]).max()
fig, ax = plt.subplots()     # Set up the matplotlib figure

sns.heatmap(corr, mask=mask, vmin=-vmax, vmax=vmax, square=True, linecolor="lightgray", linewidths=1, ax=ax)

for i in range(len(corr)):
    ax.text(i + 0.5, len(corr) - (i + 0.5), corr.columns[i], ha="center", va="center", rotation=45)
    for j in range(i + 1, len(corr)):
        s = "{:.3f}".format(corr.values[i, j])
        ax.text(j + 0.5, len(corr) - (i + 0.5), s, ha="center", va="center")
ax.axis("off")
plt.show()