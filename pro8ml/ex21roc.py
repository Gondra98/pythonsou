# ROC(Receiver Operating Characteristic) Curve는 모든 분류 
# 임계값에서 분류 모델의 성능을 보여주는 그래프로 x축이 FPR(1-특이도), y축이 TPR(민감도)인 그래프이다. 
# 즉 민감도와 특이도의 관계를 표현한 그래프이다. 
# ROC Curve는 AUC(Area Under Curve : 그래프 아래의 면적)를 이용해 모델의 성능을 평가한다. 
# AUC가 클수록 정확히 분류함을 뜻한다.

# fpr(1-특이도 : 위양성률)이 변할 때 tpr(민감도)이 어떻게 변하는지 알려 주는 곡선이다.

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)
# n_redundant : 독립변수 중 다른 독립변수의 선형조합으로 나타내는 성분수
print(x[:3], x.shape)   # (100, 2)
print(y[:3], y.shape)   # (100,)

# 산포도
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)
print('y_hat : ', y_hat[:5])    # [0 0 0 1 1]
print('real : ', y[:5])         # [1 0 0 1 0]

# Roc curve의 판별경계선 설정용 결정함수 사용
f_value = model.decision_function(x)
print('f_value : ', f_value[:10])
# [-0.28579821 -0.94089633 -4.23246754  2.80425793  0.06137063  2.88531461
#  -2.1095613  -1.76266592 -0.93092321  2.98569333]
print()
df = pd.DataFrame(np.vstack([f_value, y_hat, y]).T, columns=['f', 'y_hat', 'y'])
print(df.head())

# 모델 성능 파악
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_hat))
# [[44  4]
#  [ 8 44]]
acc = (44 + 44) / 100       # TP + TN / 전체수
recall = 44 / (44 + 4)      # TP / (TP + FN)
precision = 44 / (44 + 8)   # TP / (TP + FP)
specificity = 44 / (8 + 44) # TN / (FP + TN) 특이도
fallout = 8 / (8 + 44)      # FP / (FP + TN) 위양성율
print('acc : ', acc)
print('recall : ', recall)      # tpr : 1 근사하면 좋음
print('precision : ', precision)
print('specificity : ', specificity)
print('fallout : ', fallout)    # fpr : 0 근사하면 좋음
print('fallout : ', 1 - specificity)
# acc :  0.88
# recall :  0.9166666666666666
# precision :  0.8461538461538461
# specificity :  0.8461538461538461
# fallout :  0.15384615384615385
# fallout :  0.15384615384615385

print()

from sklearn import metrics
acc_score = metrics.accuracy_score(y, y_hat)
print('모델 정확도 : ', acc_score)  # 0.88

cl_rep = metrics.classification_report(y, y_hat)
print(cl_rep)
#               precision    recall  f1-score   support

#            0       0.85      0.92      0.88        48
#            1       0.92      0.85      0.88        52

#     accuracy                           0.88       100
#    macro avg       0.88      0.88      0.88       100
# weighted avg       0.88      0.88      0.88       100

print()

fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
print('fpr : ', fpr)
# [0.         0.         0.         0.02083333 0.02083333 0.04166667
#  0.04166667 0.10416667 0.10416667 0.14583333 0.14583333 0.27083333
#  0.27083333 0.29166667 0.29166667 0.41666667 0.41666667 0.45833333
#  0.45833333 0.47916667 0.47916667 1.        ]
print('tpr : ', tpr)
# [0.         0.01923077 0.78846154 0.78846154 0.82692308 0.82692308
#  0.84615385 0.84615385 0.88461538 0.88461538 0.90384615 0.90384615
#  0.92307692 0.92307692 0.94230769 0.94230769 0.96153846 0.96153846
#  0.98076923 0.98076923 1.         1.        ]
# thresholds : 분류결정 임계값(결정함수값)

plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0, 1], [0, 1], 'k--', label='random classifire line(AUC:0.5)')
plt.plot([fallout], [recall], 'ro', ms=6)   # 위양성률, 재현율 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve')
plt.legend()
plt.show()

print("AUC(Area Under the Curve)")
print('AUC : ', metrics.auc(fpr, tpr))  # 0.9547275641025641