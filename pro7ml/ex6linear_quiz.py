# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
#  - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#  - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#     참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.  

# 구분,지상파,종편,운동
# 1,0.9,0.7,4.2
# 2,1.2,1.0,3.8
# 3,1.2,1.3,3.5
# 4,1.9,2.0,4.0
# 5,3.3,3.9,2.5
# 6,4.1,3.9,2.0
# 7,5.8,4.1,1.3
# 8,2.8,2.1,2.4
# 9,3.8,3.1,1.3
# 10,4.8,3.1,35.0
# 11,NaN,3.5,4.0
# 12,0.9,0.7,4.2
# 13,3.0,2.0,1.8
# 14,2.2,1.5,3.5
# 15,2.0,2.0,3.5


from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 데이터프레임 생성
data = {
    '구분':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    '지상파':[0.9,1.2,1.2,1.9,3.3,4.1,5.8,2.8,3.8,4.8,np.nan,0.9,3.0,2.2,2.0],
    '종편':[0.7,1.0,1.3,2.0,3.9,3.9,4.1,2.1,3.1,3.1,3.5,0.7,2.0,1.5,2.0],
    '운동':[4.2,3.8,3.5,4.0,2.5,2.0,1.3,2.4,1.3,35.0,4.0,4.2,1.8,3.5,3.5]
}

df = pd.DataFrame(data)
print('원본 데이터')
print(df)
print()


# 결측치 처리 : 각 컬럼 평균으로 대체
df['지상파'] = df['지상파'].fillna(df['지상파'].mean())
df['종편'] = df['종편'].fillna(df['종편'].mean())
df['운동'] = df['운동'].fillna(df['운동'].mean())

print('결측치, 평균 대체 데이터')
print(df)
print()


#  이상치 제거 : 운동 10시간 초과 제거
df = df[df['운동'] <= 10]

print('이상치, 운동 10시간 초과 제거 데이터')
print(df)   # 10번 칼럼 값(35.0) 제거 확인
print()  


# ------------------------------------------------------
# 1. 지상파 시청시간 -> 운동시간 회귀모델 작성
# ------------------------------------------------------
print('1. 지상파 시청시간 -> 운동시간 회귀모델 작성')
x = df['지상파']
y = df['운동']

# (1) 단순 선형회귀
print('방법1 : scipy.stats.linregress() 사용. model 생성 X')
model1 = stats.linregress(x, y)
# print(model1)
print('기울기 : ', model1.slope)     # -0.6684
print('절편 : ', model1.intercept)   #  4.7096
print('p값 : ', model1.pvalue)       #  6.3475
# print('회귀식 : 운동 =', model1.slope, '* 지상파 +', model1.intercept)
print()

# 지상파 시청시간 입력 후 예측
new_x = float(input('지상파 시청시간 입력 : '))
print('예측값 : ', np.polyval([model1.slope, model1.intercept], np.array([new_x])))
print()


# (2) LinearRegression 사용
print('\n방법2 : LinearRegression 사용. model 생성 O')

xx = np.array(x).reshape(-1, 1)
yy = np.array(y)

model = LinearRegression()
fit_model = model.fit(xx, yy)
print('기울기(slope) : ', fit_model.coef_)      # -0.66845
print('절편(bias) : ', fit_model.intercept_)    #  4.70967

# 예측값 확인 함수
y_newpred = fit_model.predict(xx[[0]])
print('예측값1 : ', y_newpred)     # 4.10806
y_newpred2 = fit_model.predict(np.array([[new_x]]))
print('예측값2 : ', y_newpred2)    # 2.03585


# (3) ols 사용
print('\n방법3 : ols 사용. model 생성 O')
# 잔차제곱합(RSS)을 최소화하는 가중치 벡터를 행렬 미분으로 구하는 방법
import statsmodels.formula.api as smf

print(xx.ndim) 
x1 = xx.flatten()  
print(x1.ndim) 
y1 = yy

data2 = np.array([x1, y1])
df2 = pd.DataFrame(data2.T)
df2.columns = ['x1','y1']
print(df2.head(3))

model2 = smf.ols(formula="y1 ~ x1", data=df2).fit()
print(model2.summary())
print('기울기:', model2.params['x1'])        # -0.66845
print('절편:', model2.params['Intercept'])   #  4.70967

# 예측값 확인
new_df = pd.DataFrame({'x1':[new_x]})
print('예측값 : ', model2.predict(new_df))
print()

# ------------------------------------------------------
# 2. 지상파 시청시간 -> 종편 시청시간 회귀모델 작성
# ------------------------------------------------------
print('2. 지상파 시청시간 -> 종편 시청시간 회귀모델 작성')
x2 = df['지상파']
y2 = df['종편']

# (1) 단순 선형회귀
print('방법1 : scipy.stats.linregress() 사용. model 생성 X')
model2 = stats.linregress(x2, y2)
# print(model2)
print('기울기 : ', model2.slope)     # 0.7726
print('절편 : ', model2.intercept)   # 0.2951
print('p값 : ', model2.pvalue)       # 2.2838
# print('회귀식 : 종편 =', model2.slope, '* 지상파 +', model2.intercept)
print()

# 지상파 시청시간 입력 후 예측
new_x = float(input('지상파 시청시간 입력 : '))
print('예측값 : ', np.polyval([model2.slope, model2.intercept], np.array([new_x])))
print()


# (2) LinearRegression 사용
print('\n방법2 : LinearRegression 사용. model 생성 O')
from sklearn.linear_model import LinearRegression

xx = np.array(x).reshape(-1, 1)
yy = np.array(y)

model = LinearRegression()
fit_model = model.fit(xx, yy)  # 최소제곱법으로 기울기, 절편을 반환
print('기울기(slope) : ', fit_model.coef_)      # -0.66845
print('절편(bias) : ', fit_model.intercept_)    #  4.70967

# 예측값 확인 함수
y_newpred = fit_model.predict(xx[[0]])
print('예측값1 : ', y_newpred)
y_newpred2 = fit_model.predict(np.array([[new_x]]))
print('예측값2 : ', y_newpred2)


# (3) ols 사용
print('\n방법3 : ols 사용. model 생성 O')
import statsmodels.formula.api as smf

print(xx.ndim) 
x1 = xx.flatten() 
print(x1.ndim)
y1 = yy

data2 = np.array([x1, y1])
df2 = pd.DataFrame(data2.T)
df2.columns = ['x1','y1']
print(df2.head(3))

model2 = smf.ols(formula="y1 ~ x1", data=df2).fit()
print(model2.summary())
print('기울기:', model2.params['x1'])         # -0.66845 
print('절편:', model2.params['Intercept'])    #  4.70967

# 예측값 확인
new_df = pd.DataFrame({'x1':[new_x]})
print('예측값 : ', model2.predict(new_df))

