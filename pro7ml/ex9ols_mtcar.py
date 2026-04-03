# 선형회귀분석 : mtcars dataset
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars)
print(mtcars.columns)
# ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
print(mtcars.info())
# x : hp(마력수), y : mpg(연비)
# print(mtcars.corr())
print(np.corrcoef(mtcars.hp, mtcars.mpg))   # -0.77616837
print(np.corrcoef(mtcars.wt, mtcars.mpg))   # -0.86765938

# 시각화
# plt.scatter(mtcars.hp, mtcars.mpg)
# plt.xlabel('마력수')
# plt.ylabel('연비')
# plt.show()

print('단순선형회귀 ---')
result = smf.ols(formula='mpg ~ hp', data=mtcars).fit()
print(result.summary())
# yhat = -0.0682 * x + 30.0989 + err
print('마력수 110에 대한 연비 예측값 : ', -0.0682 * 110 + 30.0989)  # 22.5969
print('마력수 110에 대한 연비 예측값 : ', result.predict(pd.DataFrame({'hp':[110]})).values)    # 22.59374995


print('\n다중선형회귀 ---')
result2 = smf.ols(formula='mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary())
37.2273     -0.0318 -3.8778
print('마력수 110 + 무게 5에 대한 연비 예측값 : ', (-0.0318 * 110 ) + (-3.8778 * 5) + 37.2273)  # 14.3403
print('마력수 110 + 무게 5에 대한 연비 예측값 : ', result2.predict(pd.DataFrame({'hp':[110], 'wt':[5]})).values)  # 14.34309224

print('\n추정치 구하기 -- 차체무게를 입력해 연비 추정 ---')
result3 = smf.ols(formula='mpg ~ wt', data=mtcars).fit()
print(result3.summary())
print('결정계수 : ', result3.rsquared)
pred = result3.predict()
print('result3 연비 예측값 : ', pred[:5])

# 새로운 차체 무게로 연비 추정
mtcars.wt = float(input('차체무게 입력:'))
new_pred = result3.predict(pd.DataFrame(mtcars.wt))
print(f'차체무게 {mtcars.wt[0]}일 때 예상 연비는 {new_pred[0]}')
