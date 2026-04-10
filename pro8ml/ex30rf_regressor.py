# RandomForest는 분류, 회귀 모두 가능. sklearn 모듈은 대개 그러하다.
# 캘리포니아 주택 가격 데이터로 회귀분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

housing = fetch_california_housing(as_frame=True)
print(housing.DESCR)
print(housing.data[:2])
print(housing.target[:2])
print(housing.feature_names[:2])
df = housing.frame
print(df.head(3))
print(df.info())

x = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

rfmodel = RandomForestRegressor(n_estimators=200, random_state=42)
rfmodel.fit(x_train, y_train)

# 예측 및 평가
y_pred = rfmodel.predict(x_test)
print(f'MSE : {mean_squared_error(y_test, y_pred):.3f}')    # 0.254
print(f'R2(결정계수) : {r2_score(y_test, y_pred):.3f}')      # 0.807

# 변수 중요도 시각화
importances = rfmodel.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8, 5))
plt.bar(range(x.shape[1]), importances[indices], align='center')
plt.xticks(range(x.shape[1]), x.columns[indices], rotation=45)
plt.xlabel('feature name')
plt.ylabel('feature importances')
plt.tight_layout()
plt.show()

print('중요 변수 순위정보 저장')
ranking = pd.DataFrame({
    'feature':x.columns[indices],
    'importance':importances[indices]
})
print(ranking)
#       feature  importance
# 0      MedInc    0.525400
# 1    AveOccup    0.138819
# 2   Longitude    0.086695
# 3    Latitude    0.086512
# 4    HouseAge    0.054694
# 5    AveRooms    0.045933
# 6  Population    0.032089
# 7   AveBedrms    0.029859

# 파라미터 튜닝
from sklearn.model_selection import RandomizedSearchCV
# GridSearchCV와 달리 사용자가 지정한 범위, 분포에서 임의로 일부 혼합만 샘플링해 탐색
# 연속적 값 범위도 가능. 무작위이기 때문에 최적 조합을 못 찾을 수도 있다.

param_dist = {
    'n_estimators':[200, 400, 800],
    'max_depth':[None, 10, 20, 30],
    'min_samples_leaf':[1, 2, 4],   # 리프노드에 필요한 최소 샘플 수
    'min_samples_split':[1, 2, 4],  # 노드 분할에 필요한 최소 샘플 수
    'max_features':[None, 'sqrt', 'log2', 1.0, 0.8, 0.6]    # 분할 시 고려할 최대 특성 수
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,      # 20개의 랜덤한 파라미터 조합을 함
    scoring='r2',
    cv=3,
    random_state=42,
    verbose=1
)

search.fit(x_train,y_train) # 탐색 학습

print('best_params : ', search.best_params_)
best = search.best_estimator_
print('best_score : ', search.best_score_)
print('final R2 : ', r2_score(y_test, best.predict(x_test)))





