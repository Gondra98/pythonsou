# 최소제곱법해를 선형 행렬 방정식으로 얻기
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

x = np.array([0,1,2,3])
y = np.array([-1,0.2,0.5,2.1])
# plt.scatter(x,y)
# plt.grid(True)
# plt.show()
# plt.close()

A = np.vstack([x,np.ones(len(x))]).T
print(A)

# 본래 데이터를 선행대수학을 이용하여 직선으로 표현하기
import numpy.linalg as lin

# y = wx + b의 w와 b 구하기
w, b = lin.lstsq(A, y)[0]       # 편미분을 이용하여 최소제곱법 연산
print(w,b)                      # 기울기: 0.96, y절편: -0.9899999999999998
# 회귀식 y_hat = 0.96x - 0.9899999999999998 얻음
for i in range(4): print('x=',i,'일 때 실제값: ',y[i],' ', '예측값: ', w*i+b)

plt.scatter(x,y,marker='o',label='실제값')
plt.plot(x,w*x+b,color='r',label='최적화된 선형직선')
plt.grid(True)
plt.show()
plt.close()

# 경험하지 않은 x값에 대한 y값이 궁금하다!
print('x=1.23456 일 때 예측값: ', w*1.23456+b)
