# 전통적 방법의 선형회귀(기계학습 중 지도학습)
print('방법4 : make_regression 사용. model 생성 X')

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 예측
score_iq = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv")
print(score_iq.head(3))
print(score_iq.info())
x = score_iq.iq
y = score_iq.score
print(x[:3])
print(y[:3])

print('상관 계수 : ', np.corrcoef(x, y)[0, 1])  # 0.88222
print(score_iq[['iq', 'score']].corr())

plt.scatter(x, y)
plt.show()

# 단순 선형회귀분석 (인과관계가 있다는 가정하에 진행)
model = stats.linregress(x, y)
print(model)    # LinregressResult(slope=np.float64(0.6514309527270075), intercept=np.float64(-2.8564471221974657), ....
print('기울기 : ', model.slope)
print('절편 : ', model.intercept)
print('p값 : ', model.pvalue)

plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept, c='r')
plt.show()
# predict() 메소드를 지원하지 않음
# print('점수예측 : ', np.polyval([model.slope, model.intercept],
#                     np.array(score_iq['iq'])))

newdf = pd.DataFrame({'iq':[55,66,77,88,150]})
print('점수예측 : \n', np.polyval([model.slope, model.intercept], newdf))


