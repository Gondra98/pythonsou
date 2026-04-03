# 독립 표본 t-검정 ( independent two sample t-test)
# 실습 : 두 가지 교육방법에 따른 평균시험 점수에 대한 검정 수행 two_sample csv

# 귀무 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 없다.
# 대립 : 두 가지 교육방법에 따른 평균시험 점수에 차이가 있다.

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/two_sample.csv")
print(data.head())
# print(data.isnull().sum())
# print(data['score'].isnull().sum())
# print(data.isnull().any)

# 교육 방법별 분리
ms = data[['method', 'score']]
m1 = ms[ms['method'] == 1]      # 방법 1
m2 = ms[ms['method'] == 2]      # 방법 2
print(m1.head(3))
print(m2.head(3))

# 교육방법에서 score 만 별도 기억
score1 = m1['score']
score2 = m2['score']
print(score1.isnull().sum())    # 0
print(score2.isnull().sum())    # 2

# score2 = score2.fillna(0)   # NaN으로 0으로 대체
score2 = score2.fillna(score2.mean())   # NaN으로 평균으로 대체

# 정규성 검정
print('score1:' , stats.shapiro(score1))    # 0.36799   정규성 만족
print('score1:' , stats.shapiro(score2))    # 0.67142


# 시각화
sns.histplot(score1, kde=True)
sns.histplot(score2, kde=True, color='blue')
plt.show()


# 등분산성 검정
from scipy.stats import levene
print('등분산성 : ' , levene(score1, score2).pvalue)    # 0.456842 만족

result = stats.ttest_ind(score1, score2, equal_var=True)
print('result : ', result)
# statistic -0.19649386, pvalue 0.84505322, df 48.0
# 해석 : pvalue 0.84505322 > 0.05이므로 귀무가설 채택


