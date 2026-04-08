# [로지스틱 분류분석 문제3]
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
# 얘를 사용해도 됨   'testdata/advertisement.csv' 

# 참여 칼럼 : 
#    - Daily Time Spent on Site : 사이트 이용 시간 (분)
#    - Age : 나이,
#    - Area Income : 지역 소득,
#    - Daily Internet Usage :일별 인터넷 사용량(분),
#    - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
# 모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
# 새로운 데이터로 분류 작업을 진행해 본다.

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("advertising.csv")
print(data.head(2), data.shape) # (1000, 10)
print(data.info())

adver_data = pd.DataFrame()
adver_data = data.drop(['Ad Topic Line', 'City', 'Male', 'Country', 'Timestamp'], axis=1)
print(adver_data.info())
print(adver_data.head(3))

x = adver_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = adver_data['Clicked on Ad'].values

print(x[:3], x.shape)   # (1000, 4)
print(y[:3], y.shape)   # (1000,)

model = LogisticRegression().fit(x,y)
y_hat = model.predict(x)
print('y_hat : ', y_hat[:5])
print('real : ', y[:5]) 

# Roc curve의 판별경계선 설정용 결정함수 사용
f_value = model.decision_function(x)
print('f_value : ', f_value[:10])


print()

df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=['f', 'y_hat', 'y'])
print(df.head())