# [로지스틱 분류분석 문제2] 
# 게임, TV 시청 데이터로 안경 착용 유무를 분류하시오.
# 안경 : 값0(착용X), 값1(착용O)
# 예제 파일 : https://github.com/pykwon  ==>  bodycheck.csv
# 새로운 데이터(키보드로 입력)로 분류 확인. 스케일링X

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/bodycheck.csv")
print(data.head(2), data.shape)
#    번호  게임   신장  체중  TV시청  안경유무
# 0   1   2  146  34     2     0
# 1   2   6  169  57     3     1 (20, 6)
print(data.info())

data2 = pd.DataFrame()
data2 = data.drop(['번호', '신장', '체중'], axis=1)
print(data2.head(2))
#    게임  TV시청  안경유무
# 0   2     2     0
# 1   6     3     1


x = data2[['게임', 'TV시청']]
y = data2['안경유무'].values
print(x.shape, ' ', y.shape)    # (20, 2)   (20,)
print(x[:3], y[:3], set(map(int, y)))
# (20, 2)   (20,)
#    게임  TV시청
# 0   2     2
# 1   6     3
# 2   9     3 [0 1 1] {0, 1}

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (14, 2) (6, 2) (14,) (6,)
print(x_train[:3], ' ',x_test[:3], ' ', y_train[:3], ' ', y_test[:3])
#     게임  TV시청
# 6    5     5
# 13   5     4
# 4    4     1       게임  TV시청
# 18   1     1
# 1    6     3
# 19   0     0   [1 1 0]   [0 1 0]

model = LogisticRegression(random_state=0)
print(model)
model.fit(x_train, y_train)

# 분류 예측
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)  # [0 1 0 1 0 1]
print('실제값 : ', y_test)  # [0 1 0 1 0 1]

print(f'총 갯수:{len(y_test)}, 오류수:{(y_test != y_pred).sum()}')  
# test data 총 갯수:6, 오류수:0

print('분류 정확도 확인 1')
print(f'{accuracy_score(y_test, y_pred)}')  # 1.0

print('분류 정확도 확인 2')
con_mat = pd.crosstab(y_test, y_pred, rownames=['실제값'], colnames=['예측값'])
print(con_mat)
# 예측값  0  1
# 실제값
# 0    3  0
# 1    0  3
print((con_mat[0][0] + con_mat[1][1]) / len(y_test))    # 1.0

print('분류 정확도 확인 3')
print('test score : ', model.score(x_test, y_test))     # 1.0
print('train score : ', model.score(x_train, y_train))  # 1.0




print('-----------------------------------------------------------')

# 학습후 검증이 된 모델 저장 후 읽기
import joblib
joblib.dump(model, 'ex20quiz.pkl')
del model
read_model = joblib.load('ex20quiz.pkl')

# 게임 , tv 입력
game = int(input('게임 입력: '))
tv = int(input('TV시청 입력: '))

# DataFrame으로 생성
new_data = pd.DataFrame([[game, tv]], columns=['게임', 'TV시청'])
# 분류 결과 예측 (0: 안경X, 1: 안경O)
new_pred = read_model.predict(new_data)
print('예측 결과 : ', new_pred)

# [안경X 확률, 안경O 확률]
print(read_model.predict_proba(new_data))
