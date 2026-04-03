# Paired Samples t-test
# 실습 : 복부 수술 전 9 명의 몸무게와 복부 수술 후 몸무게 변화
baseline = [67.2, 67.4, 71.5, 77.6, 86.0, 89.1, 59.5, 81.9, 105.5]
follow_up = [62.4, 64.6, 70.4, 62.6, 80.1, 73.2, 58.2, 71.0, 101.0]

# 귀무 : 복부 수술 전 몸무게와 복부 수술 후 몸무게의 변화는 없다.
# 대립 : 복부 수술 전 몸무게와 복부 수술 후 몸무게의 변화는 있다.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

print(np.mean(baseline))
print(np.mean(follow_up))
print('평균의 차이 : ', np.mean(baseline) - np.mean(follow_up))     # 6.91111

# 시각화
plt.bar(np.arange(2), [np.mean(baseline), np.mean(follow_up)])
plt.xlim(0, 1)
plt.xlabel('수술 전후', fontdict={'fontsize':12, 'fontweight':'bold'})
plt.show()

result = stats.ttest_rel(baseline, follow_up)
print(result)
# statistic 3.6681166, pvalue 0.0063266, df 8
