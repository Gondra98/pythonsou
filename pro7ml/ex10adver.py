# 선형회귀분석 모형의 적절성 선행조건

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import seaborn as sns
import statsmodels.formula.api as smf

advdf = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Advertising.csv", usecols=[1,2,3,4])

print(advdf.head(3), advdf.shape)
#       tv  radio  newspaper  sales
# 0  230.1   37.8       69.2   22.1
# 1   44.5   39.3       45.1   10.4
# 2   17.2   45.9       69.3    9.3 (200, 4)

print(advdf.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 200 entries, 0 to 199
# Data columns (total 4 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   tv         200 non-null    float64
#  1   radio      200 non-null    float64
#  2   newspaper  200 non-null    float64
#  3   sales      200 non-null    float64
# dtypes: float64(4)
# memory usage: 6.4 KB
# None

print(advdf.corr())
#                  tv     radio  newspaper     sales
# tv         1.000000  0.054809   0.056648  0.782224
# radio      0.054809  1.000000   0.354104  0.576223
# newspaper  0.056648  0.354104   1.000000  0.228299
# sales      0.782224  0.576223   0.228299  1.000000

print()
# 단순선형회귀모델 - ols
# x:tv, y:sales
lm = smf.ols(formula='sales ~ tv', data= advdf).fit()
print(f"coeffient:{lm.params}, p-value:{lm.pvalues}, r-squared:{lm.rsquared}")
# coeffient:Intercept    7.032594
# tv           0.047537
# dtype: float64, p-value:Intercept    1.406300e-35
# tv           1.467390e-42
# dtype: float64, r-squared:0.611875050850071

# print(0.782224 ** 2)  # 0.6118743861760001

print(lm.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  sales   R-squared:                       0.612
# Model:                            OLS   Adj. R-squared:                  0.610
# Method:                 Least Squares   F-statistic:                     312.1
# Date:                Mon, 06 Apr 2026   Prob (F-statistic):           1.47e-42
# Time:                        09:59:06   Log-Likelihood:                -519.05
# No. Observations:                 200   AIC:                             1042.
# Df Residuals:                     198   BIC:                             1049.
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      7.0326      0.458     15.360      0.000       6.130       7.935
# tv             0.0475      0.003     17.668      0.000       0.042       0.053
# ==============================================================================
# Omnibus:                        0.531   Durbin-Watson:                   1.935
# Prob(Omnibus):                  0.767   Jarque-Bera (JB):                0.669
# Skew:                          -0.089   Prob(JB):                        0.716
# Kurtosis:                       2.779   Cond. No.                         338.
# ==============================================================================

print(lm.summary().tables[1])
# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      7.0326      0.458     15.360      0.000       6.130       7.935
# tv             0.0475      0.003     17.668      0.000       0.042       0.053
# ==============================================================================

print()

# 예측
x_new = pd.DataFrame({'tv':advdf.tv[:3]})
print(x_new)
#       tv
# 0  230.1
# 1   44.5
# 2   17.2
print('실제값 : ', advdf.sales[:3])
print('예측값 : ', lm.predict(x_new).values)
# 실제값 :  0    22.1
# 1    10.4
# 2     9.3
# Name: sales, dtype: float64
# 예측값 :  [17.97077451  9.14797405  7.85022376]

print('직접 계산 : ', lm.params.tv * 230.1 + lm.params.Intercept )  # 직접 계산 :  17.970774512765537


print()
# 경험하지 않은 tv 광고비에 따른 상품 판매량 예측
my_new = pd.DataFrame({'tv':[100, 350, 780]})
print('예측 상품 판매량 : ', lm.predict(my_new).values) # 예측 상품 판매량 :  [11.78625759 23.6704177  44.11117309]

# 시각화
plt.scatter(advdf.tv, advdf.sales)
plt.xlabel('tv광고비')
plt.ylabel('상품판매량')
ypred = lm.predict(advdf.tv)
plt.plot(advdf.tv, ypred, c='red')
plt.title('단순선형회귀')
plt.grid(True)
plt.show()


print('\n단순선형회귀 모델이므로 적절성 선행조건 중 잔차의 정규성, 선형성, 등분산성 확인')
# 잔차(Residual) : 실제 관측값과 모델이 예측한 값의 차이를 의미다.
# 모델이 데이터를 얼마나 잘 설명하는지 보여주는 척도.
fitted = lm.predict(advdf)      # lm.predict(advdf.tv)
residual = advdf['sales'] - fitted
print('실제값:', advdf['sales'][:5].values)
print('예측값:', fitted[:5].values)
print('잔차값 :', residual[:5].values)
print('잔차평균값 :', np.mean(residual[:5]))
# 실제값: [22.1 10.4  9.3 18.5 12.9]
# 예측값: [17.97077451  9.14797405  7.85022376 14.23439457 15.62721814]
# 잔차값 : [ 4.12922549  1.25202595  1.44977624  4.26560543 -2.72721814]
# 잔차평균값 : 1.6738829920227805

print('잔차의 정규성 : 잔차가 정규성을 따르는지 확인')
from scipy.stats import shapiro
import statsmodels.api as sm

stat, p = shapiro(residual)
print(f"통계량 : {stat}, p-value:{p}")
# 통계량 : 0.9905306561484953, p-value:0.21332551436720226 > 0.05이므로 정규성 만족
print("정규성 만족" if p > 0.05 else "정규성 위배")
# Q-Q plot으로 시각화
sm.qqplot(residual, line='s')
plt.title("Q-Q plot으로 정규성 만족 확인")
plt.show()

print("선형성 검정 : 독립변수의 변화에 종속변수도 변화하나 특정한 패턴이 있으면 안됨")
# 독립변수와 종속변수 간에 선형형태로 적절하게 모델링 되었는지 검정
from statsmodels.stats.diagnostic import linear_reset   # 선형성 확인 모듈
reset_result = linear_reset(lm, power=2, use_f=True)
print('reset_result 결과 : ', reset_result.pvalue)
print("선형성 만족" if reset_result.pvalue > 0.05 else "선형성 위배")
# reset_result 결과 :  0.055736587109527704
# 선형성 만족

# 선형성 검정 시각화
sns.regplot(x=fitted, y=residual, lowess=True, line_kws={'color':'red'})
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='grey')
plt.show()




print("등분산성 검정 : 모든 x값에서 오차의 퍼짐이 유사해야 한다")
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residual, sm.add_constant(advdf['tv']))
print(bp_test)  # (np.float64(48.037965662293594), np.float64(4.180455907755742e-12), np.float64(62.59140477151428), np.float64(1.7618451425695933e-13))
bp_stat, bp_pvalue = bp_test[0], bp_test[1]
print(f"breuschpagan test : 통계량:{bp_stat}, p-value:{bp_pvalue}")
print("등분산성 만족" if bp_pvalue > 0.05 else "등분산성 위배")
# breuschpagan test : 통계량:48.037965662293594, p-value:4.180455907755742e-12
# 등분산성 위배

print()

# 참고 : Cook's distance - 특정 데이터가 회귀모델에 얼마나 영향을 주는지 확인
# 영향력 있는 관측치(이상치)를 탐지하는 진단 방법
# 데이터가 적을 때, 이상치가 의심스러울 때, 모델 결과가 이상하게 나올 때...
from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(lm).cooks_distance # 쿡거리, 인덱스 반환

# 쿡거리가 가장 큰 5개 확인
print(cd.sort_values(ascending=False).head())
# 35     0.060494
# 178    0.056347
# 25     0.038873
# 175    0.037181
# 131    0.033895
# dtype: float64

# 쿡거리가 가장 큰(영향력이 큰) 관측치 원본 확인
print(advdf.iloc[[35, 178, 25, 175, 131]])
#         tv  radio  newspaper  sales
# 35   290.7    4.1        8.5   12.8
# 178  276.7    2.3       23.7   11.8
# 25   262.9    3.5       19.5   12.0
# 175  276.9   48.9       41.8   27.0
# 131  265.2    2.9       43.0   12.7
# 대부분tv 광고비는 매우 높으나 sales가 낮음 - 모델이 예측하기 어려운 포인트들

# 시각화
fig = sm.graphics.influence_plot(lm, alpha=0.05, criterion="cooks")
plt.show()


print("--------------------------------")
# 단순선형회귀모델 - ols
# x:tv, radio, newspapaer
# y:sales
lm_mul = smf.ols(formula='sales ~ tv + radio + newspaper', data= advdf).fit()
print(lm_mul.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                  sales   R-squared:                       0.897
# Model:                            OLS   Adj. R-squared:                  0.896
# Method:                 Least Squares   F-statistic:                     570.3
# Date:                Mon, 06 Apr 2026   Prob (F-statistic):           1.58e-96
# Time:                        11:55:37   Log-Likelihood:                -386.18
# No. Observations:                 200   AIC:                             780.4
# Df Residuals:                     196   BIC:                             793.6
# Df Model:                           3
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      2.9389      0.312      9.422      0.000       2.324       3.554
# tv             0.0458      0.001     32.809      0.000       0.043       0.049
# radio          0.1885      0.009     21.893      0.000       0.172       0.206
# newspaper     -0.0010      0.006     -0.177      0.860      -0.013       0.011
# ==============================================================================
# Omnibus:                       60.414   Durbin-Watson:                   2.084
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):              151.241
# Skew:                          -1.327   Prob(JB):                     1.44e-33
# Kurtosis:                       6.332   Cond. No.                         454.
# ==============================================================================

