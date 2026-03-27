

import pandas as pd
import scipy.stats as stats
data = [4, 6, 17, 16, 8, 9]     # 각 눈금에 대한 관측 빈도수
print(stats.chisquare(data))
# statistic(chi2) : 14.2, pvalue=0.0143876
# 판정 : 유의수준 0.05 < pvalue 0.0143876 이므로 귀무가설을 기각하고 대립가설 채택
# 검정에 사용된 데이터는 우연히 발생한 것이 아니라 필여적인 원인에 발생한 것이다.
# 이 주사위는 게임에 적합하지 않다. 라는 의견을 받아 들임

# df = n - 1 = 6 - 1 = 5
# 유의 수준 : 0.05
# 임계값? : 11.07



