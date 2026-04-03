# 집단간 차이 분석 : 평균 또는 비율 차이를 분석
# :모집단에서 추출한 표본정보를 이용하여 모집단의 다양한 특성을 과학적으로 추론할 수 있다.

# 단일 표본 t검정(One-sample t-test)
# 정규분포의 표본에 대한 기대값을 조사하는 검정방법이다.
# 예상평균값과 표본본 자료 간에 평균의 차이를 검정
# 독립변수:범주형, 종속변수:연속형
# 하나의 집단에 대한 표본 평균이 예측된 평균(모집단)과 같은지 여부를 확인

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


# 실습1 : 어느 남성 집단의 평균 키 검정
# 귀무 : 해당 집단의 평균 키가 177이다. (모수)
# 대립 : 해당 집단의 평균 키가 177이 아니다.
one_sample = [167.0, 182.7, 169.6, 176.8, 185.0]
print(np.array(one_sample).mean())  # 176.2199
result = stats.ttest_1samp(one_sample, popmean=177)     # (데이터, 예상평균값(모수의 평균))
print(result)
# statistic = -0.22139444, pvalue=0.83562821, df=4
# 해석 : 유의수준 0.05 < pvalue 0.83562821      귀무가설 채택

print()
result2 = stats.ttest_1samp(one_sample, popmean=165)
print(result2)
# statistic = 3.1846739510358337, pvalue= 0.03338566775204484
# 해석

sns.displot(one_sample, bins=10, kde=True)
plt.xlabel('data')
plt.ylabel('value')
plt.show()

