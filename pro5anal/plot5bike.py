# 자전거 공유 시스템 분석용
#  : kaggle 사이트의 Bike Sharing in Washington D.C. Dataset를 편의상 조금 변경한 dataset을 사용함

# columns : 
#  'datetime', 
#  'season'(사계절:1,2,3,4), 
#  'holiday'(공휴일(1)과 평일(0)), 
#  'workingday'(근무일(1)과 비근무일(0)), 
#  'weather'(4종류:Clear(1), Mist(2), Snow or Rain(3), Heavy Rain(4)), 
#  'temp'(섭씨온도), 'atemp'(체감온도), 
#  'humidity'(습도), 'windspeed'(풍속), 
#  'casual'(비회원 대여량), 'registered'(회원 대여량), 
#  'count'(총대여량)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import koreanize_matplotlib

plt.style.use('ggplot')     # ggplot style 사용


# EDA : 탐색적 분석
train = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv", parse_dates=['datetime'])
print(train.info())
print(train.dtypes)
print(train.shape)      # (10886, 12)
print(train.columns)
# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp']
print(train.head(3))
print(train.temp.describe())
print(train.isnull().sum())


# EDA : 탐색적 분석
pd.set_option('display.width', None)
print(train.info())
print(train.dtypes)
print(train.shape)      # (10886, 12)
print(train.columns)
# ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp']
print(train.head(3))
print(train.temp.describe())
print(train.isnull().sum())

# 년월일 시분초 별도 칼럼 추가 생성
train['year'] = train['datetime'].dt.year   # dt 연산자 사용
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
print(train.head(1))
print(train.columns)


# 대여량 시각화
figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
figure.set_size_inches(15, 5)
sns.barplot(data=train, x='year', y='count', ax=ax1)
sns.barplot(data=train, x='month', y='count', ax=ax2)
sns.barplot(data=train, x='day', y='count', ax=ax3)
sns.barplot(data=train, x='hour', y='count', ax=ax4)
ax1.set(ylabel='대여수', title='년도별 대여')
ax2.set(ylabel='월', title='월별 대여')
ax3.set(ylabel='일', title='일별 대여')
ax4.set(ylabel='시간', title='시간별 대여')
plt.show()


# boxplot
fig, axes = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(12, 10)
sns.boxplot(data=train, y="count", orient="v", ax=axes[0][0])
sns.boxplot(data=train, y="count", x="season", orient="v", ax=axes[0][1])
sns.boxplot(data=train, y="count", x="hour", orient="v", ax=axes[1][0])
sns.boxplot(data=train, y="count", x="workingday", orient="v", ax=axes[1][1])
axes[0][0].set(ylabel='대여수', title='대여')
axes[0][1].set(xlabel='계절', ylabel='대여수', title='계절별 대여량')
axes[1][0].set(xlabel='시간', ylabel='대여수', title='시간별 대여량')
axes[1][1].set(xlabel='근무일', ylabel='대여수', title='근무일에 따른 대여량')
plt.show()


# 산점도 : regplot (온도, 습도, 풍속)
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
figure.set_size_inches(12, 5)
sns.regplot(x='temp', y='count', data=train, ax=ax1)
sns.regplot(x='humidity', y='count', data=train, ax=ax2)
sns.regplot(x='windspeed', y='count', data=train, ax=ax3)
plt.show()