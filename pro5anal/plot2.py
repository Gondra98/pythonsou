# 차트 영역 객체 선언시 인터페이스
import numpy as np
import matplotlib.pyplot as plt


# 1) Matplotlib 스타일의 인터페이스
x = np.arange(10)
plt.figure()
plt.subplot(2, 1, 1)    # row, column, panel number
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()


# 2) 객체 지향 인터페이스
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()


# 차트의 종류 일부 확인
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(np.random.randn(1000), bins=100, alpha=0.9)
ax2.plot(np.random.rand(1000))
plt.show()

# 막대
data = [50, 80, 100, 90, 70]
plt.bar(range(len(data)), data)
plt.show()

err = np.random.rand(len(data))
plt.barh(range(len(data)), data, alpha=0.4, xerr=err)
plt.show()


# 원
plt.pie(data, colors=['yellow', 'blue', 'red'], explode=(0, 0.2, 0, 0.1, 0))
plt.title('Pie Chart')
plt.show()

# 박스 플롯 : 전체 데이터의 분포를 확인하기에 효과적. 이상치 확인에 도움
data = [1, 50, 80, 100, 90, 70, 300]
plt.boxplot(data)
plt.show()

# scatter(bubble chart) : 산점도 차트에 점의 크기를 동적으로 표시
n = 30
np.random.seed(0)
x = np.random.rand(n)
y = np.random.rand(n)
color = np.random.rand(n)
scale = np.pi * (np.random.rand(n) * 15) ** 2
plt.scatter(x, y, c=color, s=scale)
plt.show()

# 시계열 데이터로 선그래프
import pandas as pd
fdata = pd.DataFrame(np.random.randn(1000, 4),
            index=pd.date_range('1/1/2000', periods=1000),
            columns=list('abcd'))
print(fdata.head(3))
print(fdata.tail(3))
fdata = fdata.cumsum()
print(fdata.head(3))
plt.plot(fdata)
plt.show()

# pandas의 plot 기능
fdata.plot()
fdata.plot(kind='bar')
plt.xlabel("time")
plt.ylabel("data")
plt.show()

