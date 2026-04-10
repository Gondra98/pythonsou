# RandomForest 분류 알고리즘 
# 머신러닝에서 분류(Classification)와 회귀(Regression) 문제에 널리 사용되는 강력한 앙상블(Ensemble) 지도 학습 알고리즘입니다.
# 여러 개의 결정 트리(Decision Tree)를 생성하고, 이들의 예측 결과를 종합하여
# 최종적인 분류 결과를 도출합니다.
# 앙상블 기법 중 배깅(Bagging, Bootstrap Aggregation)
# : 복수의 샘플 데이터와 DecisionTree를 학습시키고 결과를 집계
# 참고 : 우수한 성능은 Boosting, 과적합이 걱정된다면 Bagging

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv")
print(df.head(2))
print(df.info())
print(df.shape)     # (891, 12)
print(df.isnull().any())

df = df.dropna(subset=['Pclass', 'Age', 'Sex'])
print(df.shape)     # (714, 12)

df_x = df[['Pclass', 'Age', 'Sex']] # feature
print(df_x.head(3))
#    Pclass   Age     Sex
# 0       3  22.0    male
# 1       1  38.0  female
# 2       3  26.0  female

# 전처리 -- Sex열 : Label Encoding(문자범주형->정수형)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_x.loc[:, 'Sex'] = encoder.fit_transform(df_x['Sex']) # feature
print(df_x.head(3)) # female:0, male:1
#    Pclass   Age Sex
# 0       3  22.0   1
# 1       1  38.0   0
# 2       3  26.0   0

df_y = df['Survived']   # label(class)
print(df_y.head(3))     # 1: 생존   0: 사망

print()
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=12)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
# (499, 3) (215, 3) (499,) (215,)

# 모델 생성
model = RandomForestClassifier(criterion='gini', n_estimators=500, random_state=12)
# n_estimators=결정트리수
model.fit(train_x, train_y)

pred = model.predict(test_x)
print('예측값 : ', pred[:5])
print('실제값 : ', np.array(test_y[:5]))
print('맞춘 갯수 : ', sum(test_y == pred))
print('전체 대비 맞춘 비율: ', sum(test_y == pred) / len(test_y))
print('분류 정확도 : ', accuracy_score(test_y, pred))
# 예측값 :  [1 0 0 0 0]
# 실제값 :  [1 0 0 0 1]
# 맞춘 갯수 :  178
# 전체 대비 맞춘 비율:  0.827906976744186
# 분류 정확도 :  0.827906976744186


# 교차 검증 (KFold)
cross_vali = cross_val_score(model, df_x, df_y, cv=5)
print(cross_vali)   # [0.75524476 0.8041958  0.81818182 0.83216783 0.83098592]
print('교차 검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5)) # 0.80816

print('중요 변수 확인하기 ---')
print('특성(변수) 중요도 : ', model.feature_importances_)   # [0.16172779 0.49842824 0.33984396]
# feature_importances_ : 각 특성이 예측에 기여한 정도(중요도)를 수치로 표현
# 값의 합은 1.0, 수치가 클수록 해당 변수가 불순도 감소에 더 많이 기여함.

# 시각화
import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('Feature importance Score')
plt.ylabel('Features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

print()
# 전체 변수 대상으로 중요도 확인
# Name, Ticket, Cabin : 문자형 - 바로 사용 불가 (Encoding 필요)
# PassengerId, Name : Survived와 상관없는 변수

df_imsi = df[['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch']]

# Sex 다시 인코딩 필요
df_imsi.loc[:, 'Sex'] = encoder.fit_transform(df_imsi['Sex'])

train_x, test_x, train_y, test_y = train_test_split(df_imsi, df_y, test_size=0.3, random_state=12)

model.fit(train_x, train_y)

importances = model.feature_importances_
# 컬럼명 + 중요도
feature_df = pd.DataFrame({
    'feature': df_imsi.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
print(feature_df)
#   feature  importance
# 1     Age    0.292984
# 3    Fare    0.276886
# 2     Sex    0.241788
# 0  Pclass    0.090802
# 4   SibSp    0.057529
# 5   Parch    0.040012

# 시각화
import seaborn as sns
plt.figure(figsize=(8, 5))
sns.barplot(x='importance', y='feature', data=feature_df, orient='h')
plt.xlabel('Feature importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


