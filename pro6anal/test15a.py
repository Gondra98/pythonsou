# 일원분산분석으로 평균 차이 검정
# 강남구에 있는 GS편의점 3개 지역 알바생의 급여에 대한 평균 차이 검정

# 귀무 : GS편의점 3개 지역 알바생의 급여에 대한 평균은 차이가 없다.
# 대립 : GS편의점 3개 지역 알바생의 급여에 대한 평균은 차이가 있다.

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols # 추정 및 검정, 회귀, 시계열 분석 등의 기능 제공
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import koreanize_matplotlib
import urllib.request

uri = "https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/group3.txt"
# 읽기1
# data = pd.read_csv(uri, header=None)
# print(data)     type:DataFrame

# 읽기2
data = np.genfromtxt(urllib.request.urlopen(uri), delimiter=",")
print(data, type(data))     # <class 'numpy.ndarray'>
print(data.shape)   # (22, 2)

# 세개의 집단에 월급 자료 얻기, 평균
gr1 = data[data[:, 1] == 1, 0]
gr2 = data[data[:, 1] == 2, 0]
gr3 = data[data[:, 1] == 3, 0]
print(gr1, ' ', np.mean(gr1))   # 316.625
print(gr2, ' ', np.mean(gr2))   # 256.444
print(gr3, ' ', np.mean(gr3))   # 278.0

print()
# 정규성 확인
print(stats.shapiro(gr1).pvalue)    # 0.3336
print(stats.shapiro(gr2).pvalue)    # 0.6561
print(stats.shapiro(gr3).pvalue)    # 0.8324

# 등분산성
print(stats.levene(gr1, gr2, gr3).pvalue)       # 0.0458
print(stats.bartlett(gr1, gr2, gr3).pvalue)     # 0.3508

# 데이터 퍼짐 정도 시각화
plt.boxplot([gr1, gr2, gr3], showmeans=True)
plt.show()

# 일원분산분석 방법1 : annova_lm()
df = pd.DataFrame(data=data, columns=['pay', 'group'])
print(df)
lmodel = ols('pay ~ C(group)', data=df).fit()   # C(group) : group은 범주형
print(anova_lm(lmodel, typ=1))
# p 0.043589 < 0.05 이므로 귀무 기각


# 일원분산분석 방법2 : f_oneway()
f_stat, p_val = stats.f_oneway(gr1, gr2, gr3)
print('f_stat : ', f_stat)
print('p_val : ', p_val)    # 0.043589

print()
# 사후 검정
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukResult = pairwise_tukeyhsd(endog=df.pay, groups=df.group)
print(tukResult)

# 시각화
tukResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()

# 참고
# annova_lm() : 정규성, 등분산성이 깨지면 p-value 신뢰 불가
# f_oneway() : 정규성 깨지면 welch ANOVA, 등분산성이 깨지면 stats.kruskal() 사용


