# 최근접 이웃(K-Nearest Neighbors) - breast_cancer dataset 사용

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
x = data.data   # feature
y = data.target # label (0:malignant, 1:benign)
print(x[:2], ' ', x.shape)  # (569, 30)
print(y[:2], ' ', y.shape)  # (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링 필요(∵ 거리기반 모델이므로 크기가 영향을 미침)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)  # 학습용
x_test_scaled = scaler.transform(x_test)    # 검증용

# K-NN은 K값이 중요
# k값 변화에 따른 정확도 비교로 최적의 k값 얻기
# k값이 작으면 과적합, k값이 크면 과소적합
train_acc = []
test_acc = []
k_range = range(3, 12)
for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_scaled, y_train)
    # 예측
    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)
    # 정확도
    train_acc.append(accuracy_score(y_train, y_train_pred))
    test_acc.append(accuracy_score(y_test, y_test_pred))

# 시각화
plt.figure()
plt.plot(k_range, train_acc, marker='o', label='Train acc')
plt.plot(k_range, test_acc, marker='s', label='Test acc')
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.title('knn acc comp')
plt.legend()
plt.grid()
plt.show()
# 그래프 기준으로 최적은 k=3
# test acc가 가장 높은 지점이 3 (과적합 의심)
# 4는 불안, 7 ~ 9는 안정적(실무에서는 바람직)
best_k = k_range[np.argmax(test_acc)]
print('최적의 k :', best_k) # test acc가 가장 높은 지점 : 3

# 최종 모델 작성
best_k = 9
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(x_train_scaled, y_train)
# 성능 확인
y_pred = final_model.predict(x_test_scaled)
print('final_model 정확도 : ', accuracy_score(y_test, y_pred))  # 0.9736842105263158
print('분류 리포트\n', classification_report(y_test, y_pred))
#                precision    recall  f1-score   support

#            0       1.00      0.93      0.96        42
#            1       0.96      1.00      0.98        72

#     accuracy                           0.97       114
#    macro avg       0.98      0.96      0.97       114
# weighted avg       0.97      0.97      0.97       114
print('Confusion matrix\n', confusion_matrix(y_test, y_pred))
#  [[39  3]
#  [ 0 72]]

# 새로운 자료로 예측 (그냥 기존 자료를 사용)
new_data = x[0].copy()
new_data = new_data + np.random.normal(0, 0.1, size=new_data.shape)
new_data_scaled = scaler.transform([new_data]) # 스케일링
prediction = final_model.predict(new_data_scaled)
proba = final_model.predict_proba(new_data_scaled)
print('새로운 데이터 예측 결과')
print('예측 : ', prediction[0], ', 참고 0:악성, 1:양성')
print('확률 : ', proba)
# 예측 :  0 , 참고 0:악성, 1:양성
# 확률 :  [[0.55555556 0.44444444]]



