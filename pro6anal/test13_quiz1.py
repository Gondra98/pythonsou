# [two-sample t 검정 : 문제1] 
# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
# 수집된 자료 :  
#    blue : 70 68 82 78 72 68 67 68 88 60 80
#    red : 60 65 55 58 67 59 61 68 77 66 66

import numpy as np
from scipy import stats

blue = np.array([70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80])
red = np.array([60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66])

# 귀무 : 포장지 색상에 따른 제품의 매출액에 차이가 없다.
# 대립 : 포장지 색상에 따른 제품의 매출액에 차이가 있다.

shapiro_blue = stats.shapiro(blue)
shapiro_red = stats.shapiro(red)
print(shapiro_blue)
print(shapiro_red)
# p > 0.05 정규성 만족


levene = stats.levene(blue, red)
print(f"Levene p-value: {levene.pvalue:.4f}")
# p 0.4392 > 0.05 등분산성 만족


t_stat, p_val = stats.ttest_ind(blue, red, equal_var=True)

print(f"t-통계량: {t_stat:.4f}")
print(f"p-value: {p_val:.4f}")

# 결론: p-value 0.0083 < 0.05 이므로 귀무가설을 기각
# 포장지 색상에 따른 매출액에 통계적으로 유의미한 차이가 존재

