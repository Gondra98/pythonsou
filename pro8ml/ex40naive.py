# 나이브베이즈 알고리즘을 이용한 분류 - weather.csv
import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/weather.csv")
print(df.head(2))
#          Date  MinTemp  MaxTemp  Rainfall  Sunshine  WindSpeed  Humidity  Pressure  Cloud  Temp RainToday RainTomorrow
# 0  2016-11-01      8.0     24.3       0.0       6.3         20        29    1015.0      7  23.6        No          Yes
# 1  2016-11-02     14.0     26.9       3.6       9.7         17        36    1008.4      3  25.7       Yes          Yes
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 366 entries, 0 to 365
# Data columns (total 12 columns):
#  #   Column        Non-Null Count  Dtype  
# ---  ------        --------------  -----  
#  0   Date          366 non-null    object 
#  1   MinTemp       366 non-null    float64
#  2   MaxTemp       366 non-null    float64
#  3   Rainfall      366 non-null    float64
#  4   Sunshine      363 non-null    float64
#  5   WindSpeed     366 non-null    int64
#  6   Humidity      366 non-null    int64
#  7   Pressure      366 non-null    float64
#  8   Cloud         366 non-null    int64
#  9   Temp          366 non-null    float64
#  10  RainToday     366 non-null    object
#  11  RainTomorrow  366 non-null    object
# dtypes: float64(6), int64(3), object(3)

# 전처리 작업
df = df.drop('Date', axis=1)

# 범주형 처리
df['RainToday'] = df['RainToday'].map({'Yes':1, 'No':0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})
print(df.head(2))

# 결측치 처리
df['Sunshine'] = df['Sunshine'].fillna(df['Sunshine'].mean())

x = df.drop('RainTomorrow', axis=1) # feature
y = df['RainTomorrow']  # label(class)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Naive Bayes 모델 학습
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

# 예측 및 평가
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
pred = model.predict(x_test)
print('분류 정확도 : ', accuracy_score(y_test, pred))           # 0.8783783783783784
print('confusion_matrix : \n', confusion_matrix(y_test, pred))
#  [[55  6]
#  [ 3 10]]

print()
# 교차 검증
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)
print(f'교차 검증 결과에서 각 fold:{scores}, 평균:{scores.mean()}')
# 교차 검증 결과에서 각 fold:[0.72972973 0.82191781 0.79452055 0.8630137  0.83561644], 평균:0.8089596445760829

print()
# feature 중요도 분석
# feature가 정규분포를 따른다는 가정하에 클래스별 평균
# GaussianNB의 멤버로 theta_ : 각 클래스별 feature 평균
mean_0 = model.theta_[0]    # RainTomorrow=0 경우 평균 -> 비 안오는날 평균
mean_1 = model.theta_[1]    # RainTomorrow=1 경우 평균 -> 비 오는날 평균

# 각 feature가 '비오는날 vs 비안오는날'에서 얼마나 차이가 나는가?
importance = np.abs(mean_1 - mean_0)

feat_impo = pd.DataFrame({
    'feature':x.columns,
    'importance':importance
}).sort_values(by='importance', ascending=False)
print('feature 중요도')
print(feat_impo)
#      feature  importance
# 5   Humidity   15.756059
# 6   Pressure    6.070088
# 3   Sunshine    3.698378
# 0    MinTemp    3.448954
# 7      Cloud    2.623589
# 2   Rainfall    1.151417
# 1    MaxTemp    0.745820
# 8       Temp    0.296384
# 9  RainToday    0.157575
# 4  WindSpeed    0.094734

# importance에 의한 시각화
import matplotlib.pyplot as plt
import koreanize_matplotlib
plt.figure()
plt.bar(feat_impo['feature'], feat_impo['importance'])
plt.xlabel('feature')
plt.ylabel('중요도(평균)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 새로운 데이터로 예측
print('새로운 자료로 예측')
newdata = pd.DataFrame([{
    'MinTemp':12.3,
    'MaxTemp':27.0,
    'Rainfall':0.0,
    'Sunshine':10.0,
    'WindSpeed':8.0,
    'Humidity':40,
    'Pressure':1005.0,
    'Cloud':1,
    'Temp':20.0,
    'RainToday':0
}])

newpred = model.predict(newdata)
print('예측 결과 : ', '비옴' if newpred == 1 else '비안옴')
print('확률은 ', model.predict_proba(newdata))