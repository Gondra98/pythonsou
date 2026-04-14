# SVM

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler    # 표준화
from sklearn.linear_model import LogisticRegression
# LogisticRegression : 다중 클래스(label, 종속변수)를 지원하도록 일반화 됨
# 이를 softmax regression 또는 multinorminal logistic regression이라고 부른다.

iris = datasets.load_iris()
print(iris.keys())  # ['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']
print(iris.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
print(iris.data[:3])
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]]
print(np.corrcoef(iris.data[:, 2], iris.data[:, 3])[0,1])   # 0.9628654314027961

x = iris.data[:, [2, 3]]
y = iris.target
print(x.shape, ' ', y.shape)    # (150, 2)   (150,)
print(x[:3], y[:3], set(map(int, y)))
# [[1.4 0.2]
#  [1.4 0.2]
#  [1.3 0.2]] [0 0 0] {0, 1, 2}

print('\ntrain/test split (7:3)')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     # (105, 2) (45, 2) (105,) (45,)
print(x_train[:3], ' ',x_test[:3], ' ', y_train[:3], ' ', y_test[:3])
# [[3.5 1. ]
#  [5.5 1.8]
#  [5.7 2.5]]   [[5.1 2.4]
#  [4.  1. ]
#  [1.4 0.2]]   [1 2 2]   [2 1 0]

"""
# Scaling (데이터 크기 표준화 - 최적화 과정에서 안정성, 수렴속도 향상, 과적합/과소적합 방치 등의 효과)
sc = StandardScaler()
sc.fit(x_train) # 독립변수(feature)만 표준화함
sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
print(x_train[:3])
# [[-0.05624622 -0.18650096]
#  [ 1.14902997  0.93250481]
#  [ 1.26955759  1.91163486]]
print(x_test[:3])
# [[ 0.90797473  1.77175914]
#  [ 0.24507283 -0.18650096]
#  [-1.32178623 -1.30550673]]

# 스케일링 결과 원복
ori_x_train = sc.inverse_transform(x_train)
print(ori_x_train[:3])
# [[3.5 1. ]
#  [5.5 1.8]
#  [5.7 2.5]]
# 하지만 iris dataset은 크기의 차이가 거의 없으므로 표준화는 의미없다.
"""

print('분류 모델 생성 -----------------')
from sklearn import svm

model = svm.SVC(kernel='rbf' ,C=1.0)
print(model)
model.fit(x_train, y_train)     # train으로 학습

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)
# 예측값 :  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  2 1 1 2 0 2 0 0]
print('실제값 : ', y_test)
# 실제값 :  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0
#  1 1 1 2 0 2 0 0]

print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')  
# test data 총 갯수:45, 오류수:1

print('분류 정확도 확인 1')
print(f'{accuracy_score(y_test, y_pred)}')  # 0.9777777777777777

print('분류 정확도 확인 2')
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)
# 예측값   0   1   2
# 실제값
# 0    16   0   0
# 1     0  17   1
# 2     0   1  11
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))    # 0.9777777777777777


print('분류 정확도 확인 3')
print('test score : ', model.score(x_test, y_test))     # 0.9777777777777777
print('train score : ', model.score(x_train, y_train))  # 0.9809523809523809
# test score와 train score의 차이가 크다면 과적합(overfitting) 의심.

# 학습후 검증이 된 모델 저장 후 읽기
import joblib   # pickle 보다 빠르고 대용량 지원
joblib.dump(model, 'logimodel.pkl') # sav, model, ...
del model
read_model = joblib.load('logimodel.pkl')

# 이 후에는 read_model 사용
print('새로운 값으로 예측하기')
new_data = np.array([[5.5, 2.2], [0.6, 0.3], [1.1, 0.5]])
# 주의 : 만약 표준화된 자료로 모델을 생성 했다면
# sc.fit(new_data)
# new_data = sc.transfrom(new_data)

new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred) # [2 0 0]
# 위의 결과는 softmax의 확률값 중 가장 큰 인덱스가 출력된 값이다.
# print(read_model.predict_proba(new_data))   # softmax의 출력값
# [[4.50306187e-06 5.66446262e-03 9.94331034e-01]
#  [9.99983341e-01 1.37693175e-05 2.88987896e-06]
#  [9.99983345e-01 1.37693176e-05 2.88568663e-06]]

# 시각화
# iris dataset 분류 연습용 시각화 코드
import matplotlib.pyplot as plt
import koreanize_matplotlib
from matplotlib.colors import ListedColormap

def plot_decision_regionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s', 'x', 'o', '^', 'v')      # 마커 표시 모양 5개 정의
    colors = ('r', 'b', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #print('cmap : ', cmap.colors[0], cmap.colors[1], cmap.colors[2])

    # decision surface 그리기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    # xx, yy를 ravel()를 이용해 1차원 배열로 만든 후 전치행렬로 변환하여 퍼셉트론 분류기의 
    # predict()의 인자로 입력하여 계산된 예측값을 Z로 둔다.
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)   # Z를 reshape()을 이용해 원래 배열 모양으로 복원한다.

    # X를 xx, yy가 축인 그래프 상에 cmap을 이용해 등고선을 그림
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)   
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), marker=markers[idx], label=cl)

    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c=[], linewidth=1, marker='o', s=80, label='testset')

    plt.xlabel('꽃잎 길이')
    plt.ylabel('꽃잎 너비')
    plt.legend(loc=2)
    plt.title(title)
    plt.show()

x_combined_std = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regionFunc(X=x_combined_std, y=y_combined, classifier=read_model, test_idx=range(105, 150), title='scikit-learn제공')  

