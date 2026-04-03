# 이원분산분석 (Two-way ANOVA)
# : 두 개의 독립변수(요인)가 종속변수에 미치는 영향을 분석하는 방법
#   → 각 요인의 주효과와 상호작용효과를 함께 검정

# 주효과 (Main Effect)
# : 다른 요인의 영향을 통제한 상태에서,
#   각 독립변수가 종속변수에 미치는 평균적인 영향

# 상호작용효과 (Interaction Effect) (교호작용)
# : 한 독립변수의 효과가 다른 독립변수의 수준에 따라 달라지는 현상

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns

np.set_printoptions(suppress=True, precision=4)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option("display.max_columns", None)

# 실습 1) 태아 수와 관측자 수가 태아의 머리 둘래 평균에 영향을 주는가?

# 주효과 가설 
# 귀무 : 태아 수와 태아의 머리 둘래 평균은 차이가 없다.
# 대립 : 태아 수와 태아의 머리 둘래 평균은 차이가 있다.
# 귀무 : 관측자 수와 태아의 머리 둘래 평균은 차이가 없다.
# 대립 : 관측자 수와 태아의 머리 둘래 평균은 차이가 있다.

# 교호작용 가설 
# 귀무 : 교호작용이 없다. → 태아 수와 관측자 수는 관련이 없다.
# 대립 : 교호작용이 있다. → 태아 수와 관측자 수는 관련이 있다.

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3_2.txt"
data = pd.read_csv(uri)
data.info()
# print(data.head(3))
# print(data['태아수'].unique())     # [1 2 3]
# print(data['관측자수'].unique())   # [1 2 3 4]

# 시각화
# data.boxplot(column='머리둘레', by='태아수')
# data.boxplot(column='머리둘레', by='관측자수')
# plt.show()

# 교호작용 확인 안하는 경우
# linreg = ols("머리둘레 ~ C(태아수) + C(관측자수)", data=data).fit()

# 교호작용 확인하는 경우
# linreg = ols("머리둘레 ~ C(태아수) + C(관측자수) + C(태아수):C(관측자수)", data=data).fit()
linreg = ols("머리둘레 ~ C(태아수) * C(관측자수)", data=data).fit()

result = anova_lm(linreg, typ=2)
# print(result)
#                       sum_sq      df         F    PR(>F)       결과           해석
# C(태아수)             324.0089   2.0000  2113.1014  0.0000   귀무가설 기각      태아 수에 따라 머리둘레 평균에 차이가 있다.
# C(관측자수)             1.1986   3.0000     5.2114  0.0065   귀무가설 기각      관측자 수에 따라 머리둘레 평균에 차이가 있다.
# C(태아수):C(관측자수)   0.5622   6.0000     1.2222  0.3296    귀무가설 채택      태아 수와 관측자 수는 무관하다.

# 해석 : 
# 태아 수와 관측자 수는 머리둘레(종속변수)에 유의한 영향을 미친다.
# 그러나 태아수와 관측자수 간의 상호작용 효과는 유의하지 않다.



# 실습 2) 독 종류와 응급처치 방법이 독 퍼짐 시간 평균에 영향을 주는가?

# 주효과 가설 
# 귀무 : 독 종류와 독 퍼짐 시간 평균은 차이가 없다.
# 대립 : 독 종류와 독 퍼짐 시간 평균은 차이가 있다.
# 귀무 : 응급처치 방법과 독 퍼짐 시간의 평균은 차이가 없다.
# 대립 : 응급처치 방법과 독 퍼짐 시간의 평균은 차이가 있다.

# 교호작용 가설 
# 귀무 : 교호작용이 없다. → 독 종류와 응급처치 방법은 관련이 없다.
# 대립 : 교호작용이 있다. → 독 종류와 응급처치 방법은 관련이 있다.

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/poison_treat.csv"
data = pd.read_csv(uri, index_col=0)
data.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 48 entries, 1 to 48
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype  
# ---  ------  --------------  -----  
#  0   time    48 non-null     float64
#  1   poison  48 non-null     int64  
#  2   treat   48 non-null     object 
# dtypes: float64(1), int64(1), object(1)
# memory usage: 1.5+ KB

print(data.groupby('poison').agg(len))
print(data.groupby('treat').agg(len))
print(data.groupby(['poison','treat']).agg(len))
# 요인별 레벨의 표본 수 = 4로 동일 (모든 집단별 표본수가 동일한 균형 설계가 잘됨)

result = ols("time ~ C(poison) * C(treat)", data=data).fit()
print(anova_lm(result, type=2))
#                         df  sum_sq  mean_sq       F  PR(>F)    결과        해석
# C(poison)           2.0000  1.0330   0.5165 23.2217  0.0000    귀무 기각   독 종류와 독 퍼짐 시간 평균은 차이가 있다.
# C(treat)            3.0000  0.9212   0.3071 13.8056  0.0000    귀무 기각   응급처치 종류와 독 퍼짐 시간 평균은 차이가 있다.
# C(poison):C(treat)  6.0000  0.2501   0.0417  1.8743  0.1123    귀무 채택   독 종류와 응급처치 방법은 관련이 없다.

# 사후분석
tkResult_p = pairwise_tukeyhsd(endog=data.time, groups=data.poison)
# print(tkResult_p) 
tkResult_p.plot_simultaneous(xlabel='mean of time', ylabel='poison')

tkResult_t = pairwise_tukeyhsd(endog=data.time, groups=data.treat)
# print(tkResult_t) 
tkResult_t.plot_simultaneous(xlabel='mean of time', ylabel='treat')

plt.show()
plt.close()