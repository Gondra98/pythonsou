# [XGBoost 문제] 
# kaggle.com이 제공하는 'glass datasets'          testdata 폴더 : glass.csv
# 유리 식별 데이터베이스로 여러 가지 특징들에 의해 7 가지의 label(Type)로 분리된다.
# RI	Na	Mg	Al	Si	K	Ca	Ba	Fe	Type
#                           ...
# glass.csv 파일을 읽어 분류 작업을 수행하시오.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

data = pd.read_csv("glass.csv")
print(data.head(5))
print(data.info())

x = data.drop('Type', axis=1)
y = data['Type']
print(y.unique())

mapping = {}
for i, v in enumerate(sorted(y.unique())):
    mapping[v] = i

print(mapping)  # {1:0, 2:1, 3:2, 5:3, 6:4, 7:5}
y = y.map(mapping)
print(y.unique())  # [0 1 2 3 4 5]


print(x[:3], x.shape)   # (214, 9)
print(y[:3], y.shape)   # (214,)

x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.3, random_state=42, stratify=y)

print(x_train.shape, x_test.shape)  # (149, 9) (65, 9)

xgb_clf = xgb.XGBClassifier(
    booster='gbtree',   # 'gbtree':tree기반, 'gblinear':선형모델
    max_depth=6, # 개별 결정트리 최대깊이
    n_estimators=200, # 약한 분류기의 갯수
    eval_metric='mlogloss',
    random_state = 42
)

xgb_clf.fit(x_train, y_train)


# 예측 / 평가
pred_xgb = xgb_clf.predict(x_test)
print(f'XGBClassifier acc : {accuracy_score(y_test, pred_xgb):.5f}')    # 0.78462