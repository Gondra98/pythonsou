"""
pandas 문제 5)  타이타닉 승객 데이터를 사용하여 아래의 물음에 답하시오.

타이타닉호 3D 이미지-마젤란사(심해지도 제작 업체) 홈페이지
  데이터 : https://github.com/pykwon/python/blob/master/testdata_utf8/titanic_data.csv
   titanic_data.csv 파일을 다운로드 후
   df = pd.read_csv('파일명',  header=None,,,)  

  열 구성 정보
    Survived : 0 = 사망, 1 = 생존
    Pclass : 1 = 1등석, 2 = 2등석, 3 = 3등석
    Sex : male = 남성, female = 여성
    Age : 나이
    SibSp : 타이타닉 호에 동승한 자매 / 배우자의 수
    Parch : 타이타닉 호에 동승한 부모 / 자식의 수
    Ticket : 티켓 번호
    Fare : 승객 요금
    Cabin : 방 호수
    Embarked : 탑승지, C = 셰르부르, Q = 퀸즈타운, S = 사우샘프턴

 1) 데이터프레임의 자료로 나이대(소년, 청년, 장년, 노년)에 대한 생존자수를 계산한다.
      cut() 함수 사용
     bins = [1, 20, 35, 60, 150]
      labels = ["소년", "청년", "장년", "노년"]
"""

import pandas as pd
import numpy as np

df=pd.read_csv('titanic.csv')

bins=[1,20,35,60,150]
labels=["소년","청년","장년","노년"]
df['나이대']=pd.cut(df['Age'],bins=bins,labels=labels)
result=df.groupby('나이대',observed=True)['Survived'].sum()
result=result.reset_index()
result.columns=['나이대','생존자수']
print(result)
print()



"""
  2) 성별 및 선실에 대한 자료를 이용해서 생존여부(Survived)에 대한 생존율을 피봇테이블 형태로 작성한다. 
      df.pivot_table()
     index에는 성별(Sex)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
      출력 결과 샘플1 :       

pclass	1	2	3
sex			
female	0.968085	0.921053	0.500000
male	0.368852	0.157407	0.135447

   index에는 성별(Sex) 및 나이(Age)를 사용하고, column에는 선실(Pclass) 인덱스를 사용한다.
   출력 결과 샘플2 : 위 결과물에 Age를 추가. 백분율로 표시. 소수 둘째자리까지.    예: 92.86
"""

import pandas as pd
import numpy as np

df = pd.read_csv('titanic.csv')

# 나이대 컬럼 생성
bins = [1, 20, 35, 60, 150]
labels = ["소년", "청년", "장년", "노년"]
df['나이대'] = pd.cut(df['Age'], bins=bins, labels=labels)

# 샘플1 
pivot1 = df.pivot_table(
    values='Survived',
    index='Sex',
    columns='Pclass',
    aggfunc='mean'
)
print(pivot1)
print()

# 샘플2 
pivot2 = df.pivot_table(
    values='Survived',
    index=['Sex', '나이대'],
    columns='Pclass',
    aggfunc='mean'
)
pivot2 = (pivot2 * 100).round(2)
print(pivot2)



"""
pandas 문제 6)

 https://github.com/pykwon/python/tree/master/testdata_utf8

 1) human.csv 파일을 읽어 아래와 같이 처리하시오.
     - Group이 NA인 행은 삭제
     - Career, Score 칼럼을 추출하여 데이터프레임을 작성
     - Career, Score 칼럼의 평균계산

     참고 : strip() 함수를 사용하면 주어진 문자열에서 양쪽 끝에 있는 공백과 \n 기호를 삭제시켜 준다. 
             그래서 위의 문자열에서 \n과 오른쪽에 있는 공백이 모두 사라진 것을 확인할 수 있다. 
             주의할 점은 strip() 함수는 문자열의 양 끝에 있는 공백과 \n을 제거해주는 것이지 중간에 
             있는 것까지 제거해주지 않는다.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/human.csv",skipinitialspace=True)
print(df)
df.columns = df.columns.str.strip()
print(df.dropna(subset=["Group"]))
df1 = df.dropna(subset=["Group"])
print(df1[['Career', 'Score']])
print(df1[['Career', 'Score']].mean())


"""
 2) tips.csv 파일을 읽어 아래와 같이 처리하시오.
     - 파일 정보 확인
     - 앞에서 3개의 행만 출력
     - 요약 통계량 보기
     - 흡연자, 비흡연자 수를 계산  : value_counts()
     - 요일을 가진 칼럼의 유일한 값 출력  : unique()
          결과 : ['Sun' 'Sat' 'Thur' 'Fri']
 """

df3 = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tips.csv")
print(df3.info())
print(df3.head(3))
print(df3.describe())
print(df3["smoker"].value_counts())
print(df3["day"].unique())
