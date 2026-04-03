# 세 개 이상의 모집단에 대한 가설검정 – 분산분석
# ‘분산분석’이라는 용어는 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인
# 에 의한 분산이 의미 있는 크기를 크기를 가지는지를 검정하는 것을 의미한다.
# 세 집단 이상의 평균비교에서는 독립인 두 집단의 평균 비교를 반복하여 실시할 경우에 제1종 오류가 증가하게 되어 문제가 발생한다.
# 이를 해결하기 위해 Fisher가 개발한 분산분석(ANOVA, ANalysis Of Variance)을 
# 이용하게 된다.
# 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나눠진 각 집단
# 내의 분산으로 나누고, 요인에 의한 분산이 의미 있는 크기를 가지는지를 검정한다.
# f값 = 집단간분산 / 집단내분산


# * 서로 독립인 세 집단의 평균 차이 검정
# 일원분산분석(One way ANOVA)
# 실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 
# 대상으로 실기시험을 실시. three_sample.csv
# 독립변수(범주형) : 한개의 요인: 교육방법, 방법의 종류가 3가지 (그룹이 3개)
# 종속변수(연속형) : 실기시험 평균점수
# 한개의 요인: 교육방법, 방법의 종류가 3가지(그룹이 3개)

# 귀무 : 세 가지 교육방법에 따른 시험 점수에 차이가 없다.
# 대립 : 세 가지 교육방법에 따른 시험 점수에 차이가 있다.

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols # 추정 및 검정, 회귀, 시계열 분석 등의 기능 제공
# ols (Ordinary Least Squares - 최소제곱법 회귀)

pd.set_option('display.max_columns', None)

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv")
print(data.head(3), ' ', len(data))
print(data.describe())

# 이상치(outlier)를 시각화
# import matplotlib.pyplot as plt
# plt.boxplot(data.score)
# plt.show()

data = data.query("score <= 100")
print(len(data))    # 78
print(data.describe())


# 교차표 (교육방법 별 건수)
data2 = pd.crosstab(index=data['method'], columns='count')
data2.index = ['방법1', '방법2', '방법3']
print(data2)

# 교차표(교육방법 별 만족 건수)
data3 = pd.crosstab(data['method'], data['survey'])
data3.index = ['방법1', '방법2', '방법3']
data3.columns = ['만족', '불만족']
print(data3)

print('ANOVA 검정 ---')
# F통계값을 얻기 위해 회귀분석결과(linear model)를 사용함
import statsmodels.api as sm
linreg = ols("data['score'] ~ data['method']", data=data).fit()     # 회귀분석모델 생성
result = sm.stats.anova_lm(linreg, typ=1)
print(result)
#               df(자유도)  sum_sq(제곱합)  mean_sq(제곱평균)  F(F값)    PR(>F)(p값)
# data['method']   1.0      27.980888       27.980888       0.122228    0.727597
# Residual        76.0      17398.134497    228.922822       NaN        NaN
# 해석 : p-value 0.727597 > alpha 0.05 이므로 귀무 채택
# 세 가지 

# 사후분석(Post Hoc Analysis) 하기


from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukResult = pairwise_tukeyhsd(endog=data["score"], groups=data["method"])
print(tukResult)
# Multiple Comparison of Means - Tukey HSD, FWER=0.05 
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#      1      2   0.9725 0.9702 -8.9458 10.8909  False : 유의미한 차이가 없으면 False
#      1      3   1.4904 0.9363 -8.8183  11.799  False : 유의미한 차이가 있으면 True
#      2      3   0.5179 0.9918 -9.6125 10.6483  False
# ----------------------------------------------------

# Tukey HSD 결과 시각화
import matplotlib.pyplot as plt
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
# Tukey HSD : 원래 반복 수가 동일하다는 가정하에 고안된 방법
# 집단 간 평균 차이를 정밀하게 확인 가능
# 각 집단의 표본 수의 차이가 크면 결과의 신뢰가 떨어짐


