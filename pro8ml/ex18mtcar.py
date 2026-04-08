# LogisticRegresion(로지스틱 회귀분석)
# 선형결합을 로그오즈(logit())로 해석하고, 이를 시그 모이드 함수를 통해 확률로 변환!
# 이항분류(다항도 가능), 독립변수:연속형, 종속변수:범주형
# LogisticRegresion을 근거로 뉴럴넷의 뉴련에서 사용함

# mtcars dataset 사용
import statsmodels.api as sm

mtcarsdata = sm.datasets.get_rdataset('mtcars')
print(mtcarsdata.keys())    # ['data', '__doc__', 'package', 'title', 'from_cache', 'raw_data']
mtcars = sm.datasets.get_rdataset('mtcars').data
print(mtcars.head(2))
#                 mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
# rownames
# Mazda RX4      21.0    6  160.0  110   3.9  2.620  16.46   0   1     4     4
# Mazda RX4 Wag  21.0    6  160.0  110   3.9  2.875  17.02   0   1     4     4
print(mtcars.info())
# <class 'pandas.core.frame.DataFrame'>
# Index: 32 entries, Mazda RX4 to Volvo 142E
# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   mpg     32 non-null     float64
#  1   cyl     32 non-null     int64
#  2   disp    32 non-null     float64
#  3   hp      32 non-null     int64
#  4   drat    32 non-null     float64
#  5   wt      32 non-null     float64
#  6   qsec    32 non-null     float64
#  7   vs      32 non-null     int64
#  8   am      32 non-null     int64
#  9   gear    32 non-null     int64
#  10  carb    32 non-null     int64
# dtypes: float64(5), int64(6)
# memory usage: 3.0+ KB
# None

print()
# 연비와 마력수에 따른 변속기 분류(수동, 자동)
mtcar = mtcars.loc[:, ['mpg','hp','am']]
print(mtcar.head())
#                 mpg   hp  am
# rownames
# Mazda RX4      21.0  110   1
# Mazda RX4 Wag  21.0  110   1
print(mtcar['am'].unique()) # [1(수동) 0(자동)]

# 모델 작성 방법1 : logit()
import numpy as np
import statsmodels.formula.api as smf
formula = 'am ~ hp + mpg'   # '연속형 ~ 범주형 + ...'
result = smf.logit(formula=formula, data=mtcar).fit()
print(result.summary())
#                            Logit Regression Results
# ==============================================================================
# Dep. Variable:                     am   No. Observations:                   32
# Model:                          Logit   Df Residuals:                       29
# Method:                           MLE   Df Model:                            2
# Date:                Tue, 07 Apr 2026   Pseudo R-squ.:                  0.5551
# Time:                        15:44:08   Log-Likelihood:                -9.6163
# converged:                       True   LL-Null:                       -21.615
# Covariance Type:            nonrobust   LLR p-value:                 6.153e-06
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    -33.6052     15.077     -2.229      0.026     -63.156      -4.055
# hp             0.0550      0.027      2.045      0.041       0.002       0.108
# mpg            1.2596      0.567      2.220      0.026       0.147       2.372
# ==============================================================================

# print('예측값 : ', result.predict())
pred = result.predict(mtcar[:10])
print('예측값 : ', pred.values)     #  [0.25004729 0.25004729 0.55803435 ...
print('예측값 : ', np.around(pred.values))  # np.around() 0.5 기준으로 0,1 출력
# 예측값 :  [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
print('실제값 : ', mtcar['am'][:10].values)
# 실제값 :  [1 1 1 0 0 0 0 0 0 0]
print()
print('수치에 대한 집계표(Confusion matrix, 혼돈행렬) 확인 ---')
conf_tab = result.pred_table()
print(conf_tab)
# [[16.  3.]
#  [ 3. 10.]]

# 현재 모델의 분류 정확도 1 - Confusion matrix 이용
print('분류 정확도 : ', (16 + 10) / len(mtcar))     # 0.8125
print('분류 정확도 : ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar)) # 0.8125

# 모듈로 확인 2 - Confusion matrix 이용
from sklearn.metrics import accuracy_score
pred2 = result.predict(mtcar)
print('분류 정확도 : ', accuracy_score(mtcar['am'], np.around(pred2)))  # 0.8125

print('*' * 10)
# 모델 작성 방법2 : glm() - 일반화된 선형모델
result2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()
# Binomial() : 이항분포, Gaucian : 정규분포 - 기본값
print(result2.summary())
#                  Generalized Linear Model Regression Results
# ==============================================================================
# Dep. Variable:                     am   No. Observations:                   32
# Model:                            GLM   Df Residuals:                       29
# Model Family:                Binomial   Df Model:                            2
# Link Function:                  Logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -9.6163
# Date:                Tue, 07 Apr 2026   Deviance:                       19.233
# Time:                        16:16:07   Pearson chi2:                     16.1
# No. Iterations:                     7   Pseudo R-squ. (CS):             0.5276
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    -33.6052     15.077     -2.229      0.026     -63.155      -4.055
# hp             0.0550      0.027      2.045      0.041       0.002       0.108
# mpg            1.2596      0.567      2.220      0.026       0.147       2.372
# ==============================================================================

glm_pred = result2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glm_pred.values))
print('glm 실제값 : ', mtcar['am'][:10].values)
# glm 예측값 :  [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
# glm 실제값 :  [1 1 1 0 0 0 0 0 0 0]

glm_pred2 = result2.predict(mtcar)
print('glm 모델 분류 정확도:', accuracy_score(mtcar['am'], np.around(glm_pred2)))   # 0.8125

# logit()은 변환 함수, glm()은 logit()을 포함한 전체 모델

print('새로운 값으로 분류 -----')
import pandas as pd
newdf = pd.DataFrame()
newdf['mpg'] = [10, 30, 120, 200]
newdf['hp'] = [100, 110, 80, 130]
print(newdf)
#    mpg   hp
# 0   10  100
# 1   30  110
# 2  120   80
# 3  200  130
new_pred = result2.predict(newdf)
print('예측 결과 : ', np.around(new_pred.values))   # [0. 1. 1. 1.]
print('예측 결과 : ', np.rint(new_pred.values))     # [0. 1. 1. 1.]