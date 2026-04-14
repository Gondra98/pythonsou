'''
[SVM 분류 문제] 심장병 환자 데이터를 사용하여 분류 정확도 분석 연습

https://www.kaggle.com/zhaoyingzhu/heartcsv
https://github.com/pykwon/python/tree/master/testdata_utf8         Heartcsv


Heart 데이터는 흉부외과 환자 303명을 관찰한 데이터다. 
각 환자의 나이, 성별, 검진 정보 컬럼 13개와 마지막 AHD 칼럼에 각 환자들이 심장병이 있는지 여부가 기록되어 있다. 
dataset에 대해 학습을 위한 train과 test로 구분하고 분류 모델을 만들어, 모델 객체를 호출할 경우 정확한 확률을 확인하시오. 
임의의 값을 넣어 분류 결과를 확인하시오.     
정확도가 예상보다 적게 나올 수 있음에 실망하지 말자. ㅎㅎ

feature 칼럼 : 문자 데이터 칼럼은 제외
label 칼럼 : AHD(중증 심장질환)

데이터 예)
"","Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"
"1",63,1,"typical",145,233,1,2,150,0,2.3,3,0,"fixed","No"
"2",67,1,"asymptomatic",160,286,0,2,108,1,1.5,2,3,"normal","Yes"
...

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics

# 데이터 로드
df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/Heart.csv", index_col=0)

print(df.head(3))
print(df.info())

# 결측값 제거
df = df.dropna()
print(df.info())

# 문자 컬럼 제외, label 분리
feature = df.drop(['ChestPain', 'Thal', 'AHD'], axis=1)
label = df['AHD'].map({'No':0, 'Yes':1})

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=0)

# 표준화
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# 모델 학습
svc_model = svm.SVC(C=0.01, kernel='rbf')
svc_model.fit(x_train, y_train)

pred = svc_model.predict(x_test)
print('예측값 : ', pred[:10])
print('실제값 : ', y_test[:10].values)
print('sc_score : ', metrics.accuracy_score(y_test, pred))

# 교차 검증
from sklearn import model_selection
cross_vali = model_selection.cross_val_score(svc_model, feature, label, cv=3)
print('3회 각 정확도 : ', cross_vali)
print('평균 정확도 : ', cross_vali.mean())



# 새로운 환자 데이터 예측
new_data = pd.DataFrame({
    'Age'    : [63,  45,  55],
    'Sex'    : [1,   0,   1],
    'RestBP' : [145, 130, 160],
    'Chol'   : [233, 204, 286],
    'Fbs'    : [1,   0,   0],
    'RestECG': [2,   0,   2],
    'MaxHR'  : [150, 172, 108],
    'ExAng'  : [0,   0,   1],
    'Oldpeak': [2.3, 1.4, 1.5],
    'Slope'  : [3,   1,   2],
    'Ca'     : [0,   0,   3]
})

new_pred = svc_model.predict(new_data)
for i, result in enumerate(new_pred):
    print(f'환자 {i+1} : {"심장병 있음" if result == 1 else "심장병 없음"}')





