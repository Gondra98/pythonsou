# 단순선형회귀 - iris dataset
# 상관관계가 약한 경우와 강한 경우로 회귀분석모델을 생성 후 비교

import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris.head(3), type(iris))
print(iris.iloc[:, 0:4].corr())

print("\n연습1 : 상관관계가 약한 변수를 사용 -0.117570")
result1 = smf.ols(formula='sepal_length ~ sepal_width', data=iris).fit()
print(result1.summary())
print('R squared : ', result1.rsquared) # 0.013822654
print('p-value : ', result1.pvalues.iloc[1])    # 0.151898260 > 0.05 이 모델은 유의하지 않은 모델
# 시각화
plt.scatter(iris.sepal_width, iris.sepal_length)
plt.plot(iris.sepal_width, result1.predict(), c='r')
plt.show()


print()
print()




print("\n연습2 : 상관관계가 강한 변수를 사용 0.871754")
result2 = smf.ols(formula='sepal_length ~ petal_length', data=iris).fit()
print(result2.summary())
print('R squared : ', result2.rsquared) # 0.7599546
print('p-value : ', result2.pvalues.iloc[1])    # 1.0386674194497976e-47 > 0.05 이 모델은 유의한 모델
# 시각화
plt.scatter(iris.petal_length, iris.sepal_length)
plt.plot(iris.petal_length, result2.predict(), c='b')
plt.show()


print('실제값 : ', iris.sepal_length[:10].values)   # [5.1 4.9 4.7 4.6 5.  5.4 4.6 5.  4.4 4.9]
print('예측값 : ', result2.predict()[:10])
# [4.8790946  4.8790946  4.83820238 4.91998683 4.8790946  5.00177129  4.8790946  4.91998683 4.8790946  4.91998683]
print()
# 새로운 값으로 예측
new_data = pd.DataFrame({'petal_length':[1.1, 0.5, 6.0]})
y_pred = result2.predict(new_data)
print('예측 결과 : ', y_pred.values)    # [4.75641792 4.51106455 6.76013708]


print("\n연습3 : 독립변수를 복수로 사용 - 다중선형회귀")
# result3 = smf.ols(formula='sepal_length ~ petal_length + petal_width', data=iris).fit()
column_select = "+".join(iris.columns.difference(['sepal_lenth','sepal_width','species']))
print(column_select)
result3 = smf.ols(formula='sepal_length ~ ' + column_select, data=iris).fit()
print(result3.summary())

