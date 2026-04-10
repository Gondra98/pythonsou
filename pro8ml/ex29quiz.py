# [Randomforest 문제1] 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)

# dataset은 winequality-red.csv 

# https://www.kaggle.com/sh6147782/winequalityred?select=winequality-red.csv

# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler    # 표준화
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("winequality-red.csv")

print(df.head(5), df.shape) # (1596, 12)
print(df.info())

x = df.drop('quality', axis = 1)
y = df['quality']

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=42)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) # (1117, 11) (479, 11) (1117,) (479,)

# 모델 생성
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=42)
model.fit(train_x, train_y)

pred = model.predict(test_x)
proba = model.predict_proba(test_x)
print('예측값 : ', pred[:5])
print('실제값 : ', np.array(test_y[:5]))
print('맞춘 갯수 : ', sum(test_y == pred))
print('전체 대비 맞춘 비율: ', sum(test_y == pred) / len(test_y))
print('분류 정확도 : ', accuracy_score(test_y, pred))
# 예측값 :  [5 6 5 6 6]
# 실제값 :  [5 4 5 5 6]
# 맞춘 갯수 :  337
# 전체 대비 맞춘 비율:  0.7035490605427975
# 분류 정확도 :  0.7035490605427975


# 교차 검증 (KFold)
cross_vali = cross_val_score(model, x, y, cv=5)
print(cross_vali)   # [0.53125    0.56739812 0.60815047 0.58934169 0.58934169]
print('교차 검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5)) # 0.5771

print('중요 변수 확인하기 ---')
print('특성(변수) 중요도 : ', model.feature_importances_)
# [0.07796522 0.10185498 0.07040763 0.06891127 0.08102022 0.06622831
#  0.10149321 0.09388594 0.07249419 0.10978465 0.15595438]


# 시각화
import matplotlib.pyplot as plt
n_features = x.shape[1]
plt.figure(figsize=(15,8))
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('Feature importance Score')
plt.ylabel('Features')
plt.yticks(np.arange(n_features), x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

# 새로운 와인 데이터가 들어왔을 때 품질 예측
new_wine = [[7.5, 0.5, 0.3, 2.0, 0.08, 15.0, 50.0, 0.997, 3.3, 0.6, 10.5]]
new_pred = model.predict(new_wine)
print(f"새로운 와인 샘플의 예측 품질 등급: {new_pred[0]}")
