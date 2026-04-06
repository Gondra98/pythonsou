# LinearRegression 클래스 사용 : 평가 score 정리
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression       # summary() 지원안함
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler # 정규화 클래스

# 데이터 생성
sample_size = 100
np.random.seed(1)

x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 30
print(x[:5])    # [ 16.24345364  -6.11756414  -5.28171752 -10.72968622   8.65407629]
print(y[:5])    # [ 482.83232345 -171.28184705 -154.41660926 -315.95480141  248.67317034]
print('상관 계수:', np.corrcoef(x, y)[0, 1])    # 0.9993935724865679

# 독립변수 x를 정규화 하기(0 ~ 1사이의 범위 내 자료로 변환)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print(x[:5])
print(x_scaled[:5])
# [[0.87492405]
#  [0.37658554]
#  [0.39521325]
#  [0.27379961]
#  [0.70578689]]

# 시각화
# plt.scatter(x, y)
# plt.scatter(x_scaled, y)
# plt.show()

model = LinearRegression().fit(x_scaled, y)
print('model:', model)
print('회귀계수(slope):', model.coef_)
print('절편(intercept,bias):', model.intercept_)
print('결정계수(R^2):', model.score(x_scaled, y))
y_pred = model.predict(x_scaled)    # 예측된 y
print('예측값 : ', y_pred[:5])
print('실제값 : ', y[:5])
# model: LinearRegression()
# 회귀계수(slope): [1350.4161554]
# 절편(intercept,bias): -691.1877661754081
# 결정계수(R^2): 0.9987875127274646
# 예측값 :  [ 490.32381062 -182.64057041 -157.48540955 -321.44435455  261.91825779]
# 실제값 :  [ 482.83232345 -171.28184705 -154.41660926 -315.95480141  248.67317034]


# 모델 성능 확인 함수 작성
def myRegScoreFunc(y_true, y_pred):
    # 결정계수 : 실제 관측값의 분산대비 예측값의 분산을 계산하여 데이터 예측의 정확도 성능 측정 지표
    print(f"r2_score(결정계수):{r2_score(y_true, y_pred)}")

    # 모델이 데이터의 분산을 얼마나 잘 설명하는지 나타내는 지표 (오차 분산이 작은면 점수 높음)
    print(f"explained_variance_score(설명분산점수):{explained_variance_score(y_true, y_pred)}")

    # 오차를 제곱해 평균 구함(오차가 커질수록 손실함수 값이 빠르게 증가함. 값이 작으면 모델 성능 우수)
    print(f"mean_squared_error(MSE, 평균제곱 오차):{mean_squared_error(y_true, y_pred)}")
    imsi = mean_squared_error(y_true, y_pred)   # RMSE로 변환해서 확인
    # 예측값이 실제값에서 평균적으로 얼마나 틀리는지 (오차 크기)
    print(f"mean_squared_error(RMSE, 평균제곱 오차의 제곱근):{np.sqrt(imsi)}") # 9.281592052012815 (평균적으로 약 + - 9 정도 틀림)



myRegScoreFunc(y, y_pred)   # 실제값, 예측값
# r2_score(결정계수):0.9987875127274646
# explained_variance_score(설명분산점수):0.9987875127274646
# mean_squared_error(MSE, 평균제곱 오차):86.14795101998747


print("분산이 크게 다른 x,y 값을 사용 -------")
x2 = np.random.normal(0, 1, sample_size)
y2 = np.random.normal(0, 100, sample_size) + x * 30
print(x2[:5])
print(y2[:5])    
print('상관 계수:', np.corrcoef(x2, y2)[0, 1])  # -0.05484441740006399


x2_scaled = scaler.fit_transform(x2.reshape(-1, 1))
model2 = LinearRegression().fit(x2_scaled, y2)
print('model:', model2)
print('회귀계수(slope):', model2.coef_)
print('절편(intercept,bias):', model2.intercept_)
print('결정계수(R^2):', model2.score(x2_scaled, y2))
# model: LinearRegression()
# 회귀계수(slope): [-79.32858782]
# 절편(intercept,bias): 58.55207778793868
# 결정계수(R^2): 0.003007910119952384
