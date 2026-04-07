# 참고 : 70년대 미국 보스턴 시의 주택가격을 설명한 dataset
# '''
# 회귀분석의 한 예로 scikit-learn 패키지에서 제공하는 주택가격을 예측하는 Dataset을 사용할 수 있다. 
# 이는 범죄율, 공기 오염도 등의 주거 환경 정보 등을 사용하여 70년대 미국 보스턴 시의 주택가격을 표시하고 있다.

# * 데이터 세트 특성 :
#     : 인스턴스 수 : 506
#     : 속성의 수 : 13 개의 숫자 / 범주 적 예측
#     : 중간 값 (속성 14)은 대개 대상입니다
#     : 속성 정보 (순서대로) :

# CRIM   자치시(town) 별 1인당 범죄율
# ZN 25,000   평방피트를 초과하는 거주지역의 비율
# INDUS   비소매상업지역이 점유하고 있는 토지의 비율
# CHAS   찰스강에 대한 더미변수(강의 경계에 위치한 경우는 1, 아니면 0)
# NOX   10ppm 당 농축 일산화질소
# RM   주택 1가구당 평균 방의 개수
# AGE   1940년 이전에 건축된 소유주택의 비율
# DIS   5개의 보스턴 직업센터까지의 접근성 지수
# RAD   방사형 도로까지의 접근성 지수
# TAX   10,000 달러 당 재산세율
# PTRATIO   자치시(town)별 학생/교사 비율
# B   1000(Bk-0.63)^2, 여기서 Bk는 자치시별 흑인의 비율을 말함.
# LSTAT   모집단의 하위계층의 비율(%)
# MEDV   본인 소유의 주택가격(중앙값) (단위: $1,000)

# ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

# http://archive.ics.uci.edu/ml/datasets/Housing

# 보스톤 주택 가격 데이터는 회귀를 다루는 많은 기계 학습 논문에서 사용되었다


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/housing.data", header=None, sep=r'\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
pd.set_option('display.max_columns', None)
print(df.head(2))
#       CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
# 0  0.00632  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296.0
# 1  0.02731   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242.0

#    PTRATIO      B  LSTAT  MEDV
# 0     15.3  396.9   4.98  24.0
# 1     17.8  396.9   9.14  21.6
print(df.corr())    # LSTAT(하층비율), MEDV(집값) : -0.737663

x = df[['LSTAT']].values
y = df['MEDV'].values
print(x[:3])
# [[4.98]
#  [9.14]
#  [4.03]]
print(y[:3])
# [24.  21.6 34.7]

# 단항을 통한 선형모델
model = LinearRegression()

# 다항 특성
quad = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
x_quad = quad.fit_transform(x)
x_cubic = cubic.fit_transform(x)
print(x_quad[:3])
# [[ 1.      4.98   24.8004]
#  [ 1.      9.14   83.5396]
#  [ 1.      4.03   16.2409]]
print(x_cubic[:3])
# [[  1.         4.98      24.8004   123.505992]
#  [  1.         9.14      83.5396   763.551944]
#  [  1.         4.03      16.2409    65.450827]]

# 단순회귀
model.fit(x, y)
x_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]
y_lin_fit = model.predict(x_fit)    # 그래프에 표시용
# print('y_lin_fit : ', y_lin_fit)
# y_lin_fit :  [32.9102555  31.96020614 31.01015679 30.06010744 29.11005808 28.16000873
#  27.20995937 26.25991002 25.30986067 24.35981131 23.40976196 22.45971261
#  21.50966325 20.5596139  19.60956454 18.65951519 17.70946584 16.75941648
#  15.80936713 14.85931778 13.90926842 12.95921907 12.00916971 11.05912036
#  10.10907101  9.15902165  8.2089723   7.25892295  6.30887359  5.35882424
#   4.40877488  3.45872553  2.50867618  1.55862682  0.60857747 -0.34147188
#  -1.29152124]
model_r2 = r2_score(y, model.predict(x))
print('model_r2 : ', model_r2)  # 0.5441462975864797

# 2차
model.fit(x_quad, y)
y_quad_fit = model.predict(quad.fit_transform(x_fit))    # 그래프에 표시용
q_r2 = r2_score(y, model.predict(x_quad))
print('q_r2 : ', q_r2)  # 0.6407168971636612

# 3차
model.fit(x_cubic, y)
y_cubic_fit = model.predict(cubic.fit_transform(x_fit))    # 그래프에 표시용
c_r2 = r2_score(y, model.predict(x_cubic))
print('c_r2 : ', c_r2)  # 0.6578476405895719




plt.scatter(x, y, label='초기 데이터')
plt.plot(x_fit, y_lin_fit, linestyle=':', label='linear fit(d=1), $R^2=%.2f$'%model_r2, c='b', lw=3)
plt.plot(x_fit, y_quad_fit, linestyle='-', label='quad fit(d=2), $R^2=%.2f$'%q_r2, c='r', lw=3)
plt.plot(x_fit, y_cubic_fit, linestyle='--', label='cubic fit(d=3), $R^2=%.2f$'%c_r2, c='k', lw=3)
plt.xlabel("하위계층 비율")
plt.ylabel("주택가격")
plt.legend()
plt.show()

