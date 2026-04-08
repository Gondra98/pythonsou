# Logistic Regression - 날씨 예보 (비가 올지 여부)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
print(data.head(2), data.shape)
#          Date  MinTemp  MaxTemp  Rainfall  Sunshine  WindSpeed  Humidity  Pressure  Cloud  Temp RainToday RainTomorrow
# 0  2016-11-01      8.0     24.3       0.0       6.3         20        29    1015.0      7  23.6        No          Yes
# 1  2016-11-02     14.0     26.9       3.6       9.7         17        36    1008.4      3  25.7       Yes          Yes (366, 12)
data2 = pd.DataFrame()
data2 = data.drop(['Date', 'RainToday'], axis=1)
data2['RainTomorrow'] = data2['RainTomorrow'].map({'Yes':1, 'No':0})
print(data2.head(2), data2.shape)   # (366, 10)
print(data2['RainTomorrow'].unique())   # [1 0]

# RainTomorrow : 종속변수(범주형, label, class), 나머지열 : 독립변수(feature)

print('데이터 분리 : 학습용(train data), 검증용(validation data)')
# 모델의 성능을 객관적으로 파악. 모델학습과 검증에 사용된 자료가 같다면 오버피팅(과적합) 우려 발생
train, test = train_test_split(data2, test_size=0.3, random_state=42)
print(train.shape, test.shape)
print(train.head(3))
print(test.head(3))

# 모델 생성
col_select = "+".join(train.columns.difference(['RainTomorrow']))
print(col_select)   # Cloud+Humidity+MaxTemp+MinTemp+Pressure+Rainfall+Sunshine+Temp+WindSpeed
my_formula = 'RainTomorrow ~ ' + col_select
# model = smf.glm(formula=my_formula, data=train, family=sm.families.Binomial()).fit()
model = smf.logit(formula=my_formula, data=train).fit()
print(model.summary())
#                            Logit Regression Results
# ==============================================================================
# Dep. Variable:           RainTomorrow   No. Observations:                  253
# Model:                          Logit   Df Residuals:                      243
# Method:                           MLE   Df Model:                            9
# Date:                Wed, 08 Apr 2026   Pseudo R-squ.:                  0.3995
# Time:                        11:03:54   Log-Likelihood:                -72.927
# converged:                       True   LL-Null:                       -121.45
# Covariance Type:            nonrobust   LLR p-value:                 6.232e-17
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept    219.3889     53.366      4.111      0.000     114.794     323.984
# Cloud          0.0616      0.118      0.523      0.601      -0.169       0.293
# Humidity       0.0554      0.028      1.966      0.049       0.000       0.111
# MaxTemp        0.1746      0.269      0.649      0.516      -0.353       0.702
# MinTemp       -0.1360      0.077     -1.758      0.079      -0.288       0.016
# Pressure      -0.2216      0.052     -4.276      0.000      -0.323      -0.120
# Rainfall      -0.1362      0.078     -1.737      0.082      -0.290       0.018
# Sunshine      -0.3197      0.117     -2.727      0.006      -0.550      -0.090
# Temp           0.0428      0.272      0.157      0.875      -0.489       0.575
# WindSpeed      0.0038      0.032      0.119      0.906      -0.059       0.066
# ==============================================================================
print(model.params)
# Intercept    219.388868
# Cloud          0.061599
# Humidity       0.055433
# MaxTemp        0.174591
# MinTemp       -0.136011
# Pressure      -0.221634
# Rainfall      -0.136161
# Sunshine      -0.319738
# Temp           0.042755
# WindSpeed      0.003785
print()
print('예측값:', np.rint(model.predict(test)[:5]))
# 예측값: 193    0.0
# 33     0.0
# 15     0.0
# 310    0.0
# 57     0.0
print('실제값:', test['RainTomorrow'][:5].values)   # [0 0 0 0 0]

# 분류 정확도
conf_mat = model.pred_table()
print(conf_mat)
# [[197.   9.]
#  [ 21.  26.]]
print('분류 정확도:', (conf_mat[0][0] + conf_mat[1][1])/len(train)) # 0.87109375


from sklearn.metrics import accuracy_score
pred = model.predict(test)
print('분류 정확도:', accuracy_score(test['RainTomorrow'], np.rint(pred)))  # 0.8727272727272727


