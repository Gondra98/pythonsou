# [로지스틱 분류분석 문제1]
# 문1] 소득 수준에 따른 외식 성향을 나타내고 있다. 주말 저녁에 외식을 하면 1, 외식을 하지 않으면 0으로 처리되었다. 
# 다음 데이터에 대하여 소득 수준이 외식에 영향을 미치는지 로지스틱 회귀분석을 실시하라.
# 키보드로 소득 수준(양의 정수)을 입력하면 외식 여부 분류 결과 출력하라.

# 요일,외식유무,소득수준
# 토,0,57
# 토,0,39
# 토,0,28
# 화,1,60
# 토,0,31
# 월,1,42
# 토,1,54
# 토,1,65
# 토,0,45
# 토,0,37
# 토,1,98
# 토,1,60
# 토,0,41
# 토,1,52
# 일,1,75
# 월,1,45
# 화,0,46
# 수,0,39
# 목,1,70
# 금,1,44
# 토,1,74
# 토,1,65
# 토,0,46
# 토,0,39
# 일,1,60
# 토,1,44
# 일,0,30
# 토,0,34

import pandas as pd
import statsmodels.api as sm

data = [
    ['토',0,57],
    ['토',0,39],
    ['토',0,28],
    ['화',1,60],
    ['토',0,31],
    ['월',1,42],
    ['토',1,54],
    ['토',1,65],
    ['토',0,45],
    ['토',1,98],
    ['토',1,60],
    ['토',0,41],
    ['토',1,52],
    ['일',1,75],
    ['월',1,45],
    ['화',0,46],
    ['수',0,39],
    ['목',1,70],
    ['금',1,44],
    ['토',1,74],
    ['토',1,65],
    ['토',0,46],
    ['토',0,39],
    ['일',1,60],
    ['토',1,44],
    ['일',0,30],
    ['토',0,34]
    ]

df = pd.DataFrame(data, columns=['요일','외식유무','소득수준'])
print(df)

weekend_df = df[df['요일'].isin(['토','일'])]
print(weekend_df)


# 모델 작성 방법1 : logit()
import numpy as np
import statsmodels.formula.api as smf
formula = '외식유무 ~ 소득수준'   # '연속형 ~ 범주형 + ...'
result = smf.logit(formula=formula, data=weekend_df).fit()
print(result.summary())

pred = result.predict(weekend_df)
print(pred.values)
print('예측값 : ', np.around(pred.values))
print('실제값 : ', weekend_df['외식유무'].values)

print()
print('수치에 대한 집계표(Confusion matrix, 혼돈행렬) 확인 ---')
conf_tab = result.pred_table()
print(conf_tab)

print('분류 정확도 : ', (9 + 9) / len(weekend_df))
print('분류 정확도 : ', (conf_tab[0][0] + conf_tab[1][1]) / len(weekend_df))

# 모듈로 확인 2 - Confusion matrix 이용
from sklearn.metrics import accuracy_score
pred2 = result.predict(weekend_df)
print('분류 정확도 : ', accuracy_score(weekend_df['외식유무'], np.around(pred2)))


print('*' * 10)
# 모델 작성 방법2 : glm() - 일반화된 선형모델
result2 = smf.glm(formula=formula, data=weekend_df, family=sm.families.Binomial()).fit()
print(result2.summary())

glm_pred = result2.predict(weekend_df)
print('glm 예측값 : ', np.around(glm_pred.values))
print('glm 실제값 : ', weekend_df['외식유무'].values)
print('glm 모델 분류 정확도:', accuracy_score(weekend_df['외식유무'], np.around(glm_pred)))

print('---------------------------------')


num = int(input("소득 수준 입력:"))

new_df = pd.DataFrame()
new_df['소득수준'] = [num]
print(new_df)
new_pred = result.predict(new_df)
print(new_pred.values)
print('예측 결과 : ', np.around(new_pred.values)) 


new_pred2 = result2.predict(new_df)
print(new_pred2.values)
print('예측 결과 : ', np.around(new_pred.values)) 