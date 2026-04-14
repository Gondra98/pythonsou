# kaggle의 Santander customer  satisfaction dataset 사용
# 산탄데르 은행의 고객만족 여부 분류 처리
# 클래스(label)명은 target이고 0:만족, 1:불만족

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from sklearn.model_selection import train_test_split

# pd.set_option('display.max_columns', None)

df = pd.read_csv("train_san.csv", encoding='latin-1')

# print(df.head(2))
print(df.shape)     # (76020, 371)
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 76020 entries, 0 to 76019
# Columns: 371 entries, ID to TARGET
# dtypes: float64(111), int64(260)
# memory usage: 215.2 MB
# None

# 전체 데이터에서 만족과 불만족의 비율
print(df['TARGET'].value_counts())
# TARGET
# 0    73012 만족
# 1     3008 불만족
# Name: count, dtype: int64

unsatisfied_cnt = df[df['TARGET'] == 1].TARGET.count()
total_cnt = df.TARGET.count()
print(f'불만족 비율은 {unsatisfied_cnt / total_cnt}')   # 0.0395685345961589

# print(df.describe())    # feature의 분포 확인.   var3: 이상치 발견
df['var3'].replace(-999999, 2, inplace=True)
df.drop('ID', axis=1, inplace=True)     # ID는 식별자이므로 제거
print(df.describe())

# feature / label 분리
x_features = df.iloc[:, :-1]
y_label = df.iloc[:, -1]
print('x_feature shape : ', x_features.shape)    # (76020, 369)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2, random_state=0)
train_cnt = y_train.count()
test_cnt = y_test.count()
print(x_train.shape, x_test.shape)
print('학습데이터 레이블 값 분포 비율:', y_train.value_counts() / train_cnt)
print('검증데이터 레이블 값 분포 비율:', y_test.value_counts() / test_cnt)
# 학습데이터 레이블 값 분포 비율: TARGET
# 0    0.960964
# 1    0.039036
# Name: count, dtype: float64
# 검증데이터 레이블 값 분포 비율: TARGET
# 0    0.9583
# 1    0.0417
# Name: count, dtype: float64


# 수정 후
xgb_clf = XGBClassifier(n_estimators=5, random_state=12, early_stopping_rounds=10, eval_metric='auc')
xgb_clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1])
print(f'xgb_roc_score : {xgb_roc_score:.5f}')   # 0.83431

pred = xgb_clf.predict(x_test)
print('예측값 : ', pred[:5])
print('실제값 : ', y_test[:5].values)
from sklearn import metrics
print('분류 정확도 : ', metrics.accuracy_score(y_test, pred))   # 0.9583004472507235

print()
# 최적의 파라미터 구하기
params = {'max_depth':[5, 7], 'min_child_weight':[1, 3], 'colsample_bytree':[0.5, 0.75]}
# max_depth : 트리 깊이, min_child_weight: 관측치 가중치합 최소, colsample_bytree: 피처비율
gridcv = GridSearchCV(xgb_clf, param_grid=params)
gridcv.fit(x_train, y_train, eval_set=[(x_test, y_test)])
print('gridcv 최적의 파라미터 : ', gridcv.best_params_)
gridcv_roc_score = roc_auc_score(y_test, gridcv.predict_proba(x_test)[:, 1], average='macro')
# 다중 클래스 분류 모델의 성능을 평가하는 지표
# 매크로는 클래스별 점수를 동등하게 평균내어 작은 클래스 성능을 중요시할 때,
# 마이크로는 전체 데이터의 개별 정답률을 기반으로 하여 데이터 불균형이 심할 때 사용
print(f'gridcv_roc_score : {gridcv_roc_score:.5f}')   # 0.83870
# gridcv 최적의 파라미터 :  {'colsample_bytree': 0.75, 'max_depth': 5, 'min_child_weight': 3}

print() # 위 파라미터로 모델 생성
xgb_clf2 = XGBClassifier(n_estimators=5, random_state=12, max_depth=5, min_child_weight=3, colsample_bytree=0.75)

xgb_clf2.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)])
xgb_roc_score2 = roc_auc_score(y_test, xgb_clf2.predict_proba(x_test)[:, 1])
print(f'xgb_roc_score2 : {xgb_roc_score2:.5f}')   # 0.83825

pred2 = xgb_clf2.predict(x_test)
print('예측값 : ', pred2[:5])
print('실제값 : ', y_test[:5].values)
print('분류 정확도 : ', metrics.accuracy_score(y_test, pred2))  # 0.9583662194159431

# 중요 피처 시각화
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(xgb_clf2, ax=ax, max_num_features=20)
plt.show()









