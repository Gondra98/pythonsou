# 주어진 자료 내 tv,radio,newspaper 간의 상관관계를 파악하시오. 
# sales와 관계를 알기 위해 sales에 상관 관계를 정렬한 후 TV, radio, newspaper에 대한 영향을 해석하시오.
# 이들의 관계를 heatmap 그래프로 표현하시오.

uri="https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv"

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

data = pd.read_csv(uri)
data = data.set_index('no')
data.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 200 entries, 1 to 200
# Data columns (total 4 columns):
#  #   Column     Non-Null Count  Dtype  
# ---  ------     --------------  -----  
#  0   tv         200 non-null    float64
#  1   radio      200 non-null    float64
#  2   newspaper  200 non-null    float64
#  3   sales      200 non-null    float64
# dtypes: float64(4)

# print(stats.shapiro(data.tv).pvalue)            # 1.692e-06     정규성 위배 
# print(stats.shapiro(data.radio).pvalue)         # 5.197e-07
# print(stats.shapiro(data.newspaper).pvalue)     # 1.127e-07
# print(stats.shapiro(data.sales).pvalue)         # 0.002

co_re = data.corr(method='spearman')
# print(co_re)
#                  tv     radio  newspaper     sales
# tv         1.000000  0.056123   0.050840  0.800614
# radio      0.056123  1.000000   0.316979  0.554304
# newspaper  0.050840  0.316979   1.000000  0.194922
# sales      0.800614  0.554304   0.194922  1.000000

# print(co_re['sales'].sort_values(ascending=False))
# sales        1.000000
# tv           0.800614     > 0.3   양적 상관관계
# radio        0.554304     > 0.3   양적 상관관계
# newspaper    0.194922             상관관계 없음

sns.heatmap(co_re, annot=True)
plt.show()
plt.close()