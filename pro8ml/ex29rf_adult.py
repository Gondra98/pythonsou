# RandomForest 분류알고리즘 - adult dataset (성인 소득 예측 자료)
# 연봉이 50K(약 5만달러) 이상인지 예측(이진 분류)

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline   # 전처리 + 모델을 하나로 묶어서 실행
from sklearn.compose import ColumnTransformer   # 컬럼별 전처리를 다르게 적용
from sklearn.impute import SimpleImputer    # 결측치 처리
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# data
adult = fetch_openml(name='adult', version=2, as_frame=True)
print(type(adult))  # <class 'sklearn.utils._bunch.Bunch'>
df = adult.frame
pd.set_option('display.max_columns', None)
print(df.head(5))
print(df.shape)     # (48842, 15)
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 48842 entries, 0 to 48841
# Data columns (total 15 columns):
#  #   Column          Non-Null Count  Dtype
# ---  ------          --------------  -----
#  0   age             48842 non-null  int64
#  1   workclass       46043 non-null  category
#  2   fnlwgt          48842 non-null  int64
#  3   education       48842 non-null  category
#  4   education-num   48842 non-null  int64
#  5   marital-status  48842 non-null  category
#  6   occupation      46033 non-null  category
#  7   relationship    48842 non-null  category
#  8   race            48842 non-null  category
#  9   sex             48842 non-null  category
#  10  capital-gain    48842 non-null  int64
#  11  capital-loss    48842 non-null  int64
#  12  hours-per-week  48842 non-null  int64
#  13  native-country  47985 non-null  category
#  14  class           48842 non-null  category
# dtypes: category(9), int64(6)
# memory usage: 2.7 MB
# None

# target 변환(인코딩) : class(연봉) >50K -> 1, <=50K -> 0
df['class'] = df['class'].apply(lambda x:1 if '>50K' in x else 0)
print(df.head(4))
#    hours-per-week native-country class
# 0              40  United-States     0
# 1              50  United-States     0
# 2              40  United-States     1
# 3              40  United-States     1
print(set(df['class'])) # {0, 1}

x = df.drop('class', axis = 1)  # feature
y = df['class']     # label
print(x.info())

# 컬럼 분리 : 숫자형, 범주형
num_cols = x.select_dtypes(include=['int64', 'float64']).columns    # 숫자형 칼럼만 선언
cat_cols = x.select_dtypes(include=['category', 'object']).columns    # 범주형 칼럼만 선언

# 전처리 파이프라인(숫자형)
num_pipeline = Pipeline([    # 처리항목들을 연결해 연속적으로 실행
    ('imputer', SimpleImputer(strategy='median')),   # 결측치 -> 중앙값으로 채우기
    ('scaler', StandardScaler())    # 표준화(평균0, 표준편차1)
])

# 전처리 파이프라인(범주형)
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),    # 결측치 -> 최빈값으로 채우기
    ('onehot', OneHotEncoder(handle_unknown='ignore'))          # 범주형 -> One-hot 인코딩
])

# 컬럼별 전처리 결합
# 숫자:스케일링, 문자:인코딩 -- 각각 다르게 처리해야 함
preprocess = ColumnTransformer([
    ('num', num_pipeline, num_cols), # 숫자형 칼럼에 num_pipeline 적용
    ('cat', cat_pipeline, cat_cols) # 범주형 칼럼에 cat_pipeline 적용
])

# 전체 파이프라인 (전처리 + 모델)
pipeline = Pipeline([
    ('prep', preprocess),   # 전처리 단계
    ('model', RandomForestClassifier(random_state=12))  # 모델
])

# train / test split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=12, stratify=y)

# 하이퍼파라미터 튜닝 범위 설정
param_grid = {
    'model__n_estimators':[100, 200],    # 트리 갯수
    'model__max_depth':[5, 10, None],    # 트리 깊이
    'model__class_weight':[None, 'balanced'] # 클래스 불균형 보정
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12)

grid = GridSearchCV(
    pipeline,       # 전체 파이프라인 사용
    param_grid,     # 탐색할 파라미터
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1   # 모든 CPU 사용
)

grid.fit(train_x, train_y)      # 전처리 + 최적의 파라미터 탐색 + 학습 수행
print('최적의 파라미터 : ', grid.best_params_)  # {'model__class_weight': None, 'model__max_depth': 10, 'model__n_estimators': 200}
print('최적의 모델 : ', grid.best_estimator_)
# 최적의 모델 :  Pipeline(steps=[('prep',
#                  ColumnTransformer(transformers=[('num',
#                                                   Pipeline(steps=[('imputer',
#                                                                    SimpleImputer(strategy='median')),
#                                                                   ('scaler',
#                                                                    StandardScaler())]),
#                                                   Index(['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
#        'hours-per-week'],
#       dtype='object')),
#                                                  ('cat',
#                                                   Pipeline(steps=[('imputer',
#                                                                    SimpleImputer(strategy='most_frequent')),
#                                                                   ('onehot',
#                                                                    OneHotEncoder(handle_unknown='ignore'))]),
#                                                   Index(['workclass', 'education', 'marital-status', 'occupation',
#        'relationship', 'race', 'sex', 'native-country'],
#       dtype='object'))])),
#                 ('model',
#                  RandomForestClassifier(max_depth=10, n_estimators=200,
#                                         random_state=12))])

print()

# 예측
pred = grid.predict(test_x)
proba = grid.predict_proba(test_x)[:, 1]    # 클래스 1에 대한 확률값

# 평가
print('정확도 : ', accuracy_score(test_y, pred))    # 0.8573670920630587
print('roc_auc 점수 : ', roc_auc_score(test_y, proba))  # 0.9105996302791954
print('classification_report : \n', classification_report(test_y, pred))
#                precision    recall  f1-score   support

#            0       0.87      0.95      0.91     11147
#            1       0.79      0.55      0.65      3506

#     accuracy                           0.86     14653
#    macro avg       0.83      0.75      0.78     14653
# weighted avg       0.85      0.86      0.85     14653


