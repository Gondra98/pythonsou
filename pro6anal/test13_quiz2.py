# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
# 수집된 자료 :  
#   남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
#   여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4

import numpy as np
from scipy import stats

men_pop = np.array([0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8])
women_pop = np.array([1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4])

men_sample = np.random.choice(men_pop, 15, replace=False)
women_sample = np.random.choice(women_pop, 15, replace=False)


# 귀무 : 남녀 간 콜레스테롤 양에 차이가 없다.
# 대립 : 남녀 간 콜레스테롤 양에 차이가 있다.

shapiro_men = stats.shapiro(men_sample)
shapiro_women = stats.shapiro(women_sample)
print("--- 2. 정규성 검정 결과 ---")
print(f"남자 p-value: {shapiro_men.pvalue:.4f}")
print(f"여자 p-value: {shapiro_women.pvalue:.4f}")

if shapiro_men.pvalue > 0.05 and shapiro_women.pvalue > 0.05:
    # 등분산성 확인
    levene = stats.levene(men_sample, women_sample)
    is_equal = levene.pvalue > 0.05
    t_stat, p_val = stats.ttest_ind(men_sample, women_sample, equal_var=is_equal)
    test_name = "독립표본 t-검정"

else:
    # Mann-Whitney U 검정 수행
    u_stat, p_val = stats.mannwhitneyu(men_sample, women_sample, alternative='two-sided')
    test_name = "Mann-Whitney U 검정"


print(f"분석 방법: {test_name}")
print(f"p-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("\n결론: p-value가 0.05보다 작으므로 귀무가설을 기각")
    print("남녀 간 콜레스테롤 양에 통계적으로 유의미한 차이가 존재")
else:
    print("\n결론: p-value가 0.05보다 크므로 귀무가설을 채택")
    print("남녀 간 콜레스테롤 양에 통계적으로 유의미한 차이가 있다고 볼 수 없음.")