"""
pandas 문제 1)
  a) 표준정규분포를 따르는 9 X 4 형태의 DataFrame을 생성하시오. 
     np.random.randn(9, 4)
  b) a에서 생성한 DataFrame의 칼럼 이름을 - No1, No2, No3, No4로 지정하시오
  c) 각 컬럼의 평균을 구하시오. mean() 함수와 axis 속성 사용
"""
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

data = np.random.randn(9, 4)
print(data)


frame = pd.DataFrame(data, columns=['No1', 'No2', 'No3', 'No4'])
print(frame)

print('컬럼 평균:')
print(frame.mean(axis=0))


"""
pandas 문제 2)
a) DataFrame으로 위와 같은 자료를 만드시오. colume(열) name은 numbers, row(행) name은 a~d이고 값은 10~40.
b) c row의 값을 가져오시오.
c) a, d row들의 값을 가져오시오.
d) numbers의 합을 구하시오.
e) numbers의 값들을 각각 제곱하시오. 아래 결과가 나와야 함.
f) floats 라는 이름의 칼럼을 추가하시오. 값은 1.5, 2.5, 3.5, 4.5    아래 결과가 나와야 함.
g) names라는 이름의 다음과 같은 칼럼을 위의 결과에 또 추가하시오. Series 클래스 사용.
"""
s1 = Series([10,20,30,40], index=['a','b','c','d'])
print(s1)
frame2 = pd.DataFrame(s1, columns=['name'])
print(frame2)
print(frame2.loc['c'])
print(frame2.loc['a', 'name'], frame2.loc['d', 'name'])
print("numbers의 합 : ",frame2.sum(axis=0))



