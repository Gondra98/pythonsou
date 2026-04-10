# 과적합(Overfitting) 방지 목적 : 
# train-test split : 일반화 성능 향상 
# K-Fold : 안정적 평가 
# GridSearchCV : 최적의 하이퍼파라미터 검색

# iris dataset 사용

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
print(iris.keys())

train_data = iris.data
train_label = iris.target
print(train_data[:3])
print(train_label[:3])

# 분류모델 작성
dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_data, train_label)     # 모든 데이터를 학습에 참여
pred = dt_clf.predict(train_data)   # 학습데이터로 검증(예측)
print('예측값 : ', pred)
print('실제값 : ', train_label)
print('분류 정확도 : ', accuracy_score(train_label, pred))  # 1.0   과적합 의심

print('\n과적합 방지 목적의 처리 1 - train/test split')
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=121)

dt_clf.fit(x_train, y_train)      # train data로 학습
pred2 = dt_clf.predict(x_test)    # test data로 예측
print('예측값 : ', pred2)
print('실제값 : ', y_test)
print('분류 정확도 : ', accuracy_score(y_test, pred2))  # 0.9555555555555556    효과:과적합 여부

print('\n과적합 방지 목적의 처리 2 - 교차검증(cross validation)')
# train data를 분할해 학습과 평가를 병행하는 방법 : K-Fold가 가장 일반적
from sklearn.model_selection import KFold
import numpy as np
features = iris.data
label = iris.target
dt_clf2 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=12)

kfold = KFold(n_splits=5)   # k:5회 접기
cv_acc = []
print('iris shape : ', features.shape)  # 150 by 4
# KFold 학습시 전체 150행이 학습데이터(4/5, 120개), 검증데이터(1/5, 50개)로 분할되어 학습함

n_iter = 0
# KFold 객체의 split()을 호출하면 Fold 별 학습용, 검증용 테스트의 행인덱스를 array로 변환
for train_index, test_index in kfold.split(features):
    # print('n_iter(반복수) : ', n_iter)
    # print('train_index : ', train_index)
    # print('test_index : ', test_index)
    # n_iter += 1
    xtrain, xtest = features[train_index], features[test_index]
    ytrain, ytest = label[train_index], label[test_index]
    # 학습 및 예측
    dt_clf2.fit(xtrain, ytrain)     # train으로 학습
    pred = dt_clf2.predict(xtest)   # test로 검증
    n_iter += 1
    # 반복할 때 마다 정확도 출력
    acc = np.round(accuracy_score(ytest, pred), 5)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print(f'반복수:{n_iter}, 교차검증 정확도:{acc}, 학습데이터크기:{train_size}, 검증데이터크기:{test_size}')

    print(f'반복수:{n_iter}, 검증데이터 인덱스:{test_index}')
    cv_acc.append(acc)

print('cv_acc : ', np.array(cv_acc).astype(int))
print('평균 검증 정확도 : ', np.mean(cv_acc))
# 반복수:1, 교차검증 정확도:1.0, 학습데이터크기:120, 검증데이터크기:30
# 반복수:1, 검증데이터 인덱스:[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29]
# 반복수:2, 교차검증 정확도:0.96667, 학습데이터크기:120, 검증데이터크기:30
# 반복수:2, 검증데이터 인덱스:[30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53
#  54 55 56 57 58 59]
# 반복수:3, 교차검증 정확도:0.83333, 학습데이터크기:120, 검증데이터크기:30
# 반복수:3, 검증데이터 인덱스:[60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83
#  84 85 86 87 88 89]
# 반복수:4, 교차검증 정확도:0.93333, 학습데이터크기:120, 검증데이터크기:30
# 반복수:4, 검증데이터 인덱스:[ 90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
#  108 109 110 111 112 113 114 115 116 117 118 119]
# 반복수:5, 교차검증 정확도:0.73333, 학습데이터크기:120, 검증데이터크기:30
# 반복수:5, 검증데이터 인덱스:[120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137
#  138 139 140 141 142 143 144 145 146 147 148 149]
# cv_acc :  [1 0 0 0 0]
# 평균 검증 정확도 :  0.8933320000000002



# 참고 : StratifiedKFold - 불균형한 분포도를 가진 레이블 데이터 집합을 처리하기 위한 KFold 방식
# 예를 들어 대출 사기 데이터인 경우 대부분은 정상, 사기 레이블은 극히 일부임
# from sklearn.model_selection import StratifiedKFold

print('\n과적합 방지 목적의 처리 2-1 - 교차검증 단순화')
# cross_val_score를 이용해 교차검증을 간단히 처리 가능
from sklearn.model_selection import cross_val_score     # 내부적으로 KFold 사용
data = iris.data
label = iris.target

score = cross_val_score(dt_clf2, data, label, scoring='accuracy', cv=5)
print('교차 검증별 정확도:', np.round(score, 3))                # [0.967 0.967 0.933 0.933 1.   ]
print('평균 검증 정확도 : ', np.round(np.mean(score), 3))       # 0.96



print('\n과적합 방지 목적의 처리 3 - GridSearchCV')
# 과적합 방지 간접 방법
# 최적의 파라미터 찾기(내부적으로 KFold 사용해 과적합을 줄이는데 도움을 준다)
from sklearn.model_selection import GridSearchCV
# 연습용으로 일부 파라미터만 사용 : max_depth, 
# min_samples_split : 노드 분할을 위한 최소한의 샘플수로 과적합 제어

parameters = {'max_depth':[1, 2, 3], 'min_samples_split':[2, 3]}

grid_dtree = GridSearchCV(estimator=dt_clf2, param_grid=parameters, cv=3, refit=True)

grid_dtree.fit(x_train, y_train)    
# 내부적으로 복수 개의 모형을 생성하고, 이를 실행시켜 최적의 파라미터를 찾아줌
# grid_dtree.cv_results_ : best_score_, best_params_, best_estimator_, grid_score_ ...

import pandas as pd
pd.set_option('display.max_columns', None)
scores_df = pd.DataFrame(grid_dtree.cv_results_)
print(scores_df)
#    mean_fit_time  std_fit_time  mean_score_time  std_score_time  \
# 0       0.000532      0.000025         0.000291    1.080539e-05
# 1       0.000520      0.000016         0.000304    1.398542e-05
# 2       0.000514      0.000014         0.000288    1.391887e-05
# 3       0.000515      0.000008         0.000275    1.123916e-07
# 4       0.000514      0.000010         0.000279    4.328171e-06
# 5       0.000513      0.000010         0.000277    8.558751e-06

#    param_max_depth  param_min_samples_split  \
# 0                1                        2
# 1                1                        3
# 2                2                        2
# 3                2                        3
# 4                3                        2
# 5                3                        3

#                                      params  split0_test_score  \
# 0  {'max_depth': 1, 'min_samples_split': 2}           0.657143
# 1  {'max_depth': 1, 'min_samples_split': 3}           0.657143
# 2  {'max_depth': 2, 'min_samples_split': 2}           0.942857
# 3  {'max_depth': 2, 'min_samples_split': 3}           0.942857
# 4  {'max_depth': 3, 'min_samples_split': 2}           0.971429
# 5  {'max_depth': 3, 'min_samples_split': 3}           0.971429

#    split1_test_score  split2_test_score  mean_test_score  std_test_score  \
# 0           0.657143           0.657143         0.657143        0.000000
# 1           0.657143           0.657143         0.657143        0.000000
# 2           0.914286           0.942857         0.933333        0.013469
# 3           0.914286           0.942857         0.933333        0.013469
# 4           0.914286           0.942857         0.942857        0.023328
# 5           0.914286           0.942857         0.942857        0.023328

#    rank_test_score
# 0                5
# 1                5
# 2                3
# 3                3
# 4                1
# 5                1

print('GridSearchCV 최적 파라미터 : ', grid_dtree.best_params_) # {'max_depth': 3, 'min_samples_split': 2}
print('GridSearchCV 최적 정확도 : ', grid_dtree.best_score_)    # 0.9428571428571427

# 최적의 모델
bestmodel = grid_dtree.best_estimator_  # 최적의 파라미터로 모델 생성
print(bestmodel)
best_pred = bestmodel.predict(x_test)
print('예측 결과 : ', best_pred)
print('정확도 : ', accuracy_score(y_test, best_pred))
# 예측 결과 :  [1 2 1 0 0 1 1 1 1 2 2 1 1 0 0 2 1 0 2 0 2 2 1 1 1 1 0 0 2 2 1 2 0 0 1 2 0
#  0 0 2 2 2 2 0 1]
# 정확도 :  0.9555555555555556

# 과적합 방지 기타 : 불필요한 변수 제거, 정규화(L1, L2), 데이터양 증가, 조기종료 ...





