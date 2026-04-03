# Paired Samples t-test(대응표본 t검정, 동일집단표본 t검정, 쌍체 t검정)
# 하나의 집단에 대해 독립변수를 적용하기 전과 후의 종속변수(평균)의 수준을 측정하고
# 이들의 차이가 통계적으로 유의한지를 분석하는 방법
# 동일한 관찰 대상으로 처리 이전과 이후를 1:1 대응시킨 검정 방법
# 집단 간 비교가 아니므로 등분산 가정을 할 필요는 없다.
# 예 : 광고 전후의 상품 선호도 측정, 투자 대비 상품 판매량 ...

# 실습 1: 3 강의실 학생들을 대상으로 특강이 시험 점수에 영향을 주었는가?
# 귀무 : 특강 전후의 시험 점수는 차이가 없다.
# 대립 : 특강 전후의 시험 점수는 차이가 있다.

import numpy as np
import scipy.stats as stats

np.random.seed(123)
x1 = np.random.normal(75, 10, 100)
x2 = np.random.normal(80, 10, 100)

# 정규성 확인
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(x1, kde=True)
sns.displot(x2, kde=True)
plt.show()

print(stats.shapiro(x1).pvalue)     # 0.27487044 > 0.05
print(stats.shapiro(x2).pvalue)     # 0.10214157 > 0.05

# 대응표본 t검정
print(stats.ttest_rel(x1, x2))
# statistic -3.00310270, pvalue 0.00338379, df 99
