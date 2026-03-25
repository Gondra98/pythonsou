# a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
#      - DataFrame의 자료를 파일로 저장
#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
#      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
#      - 연봉 상위 20% 직원 출력  : quantile()
#      - SQL로 1차 필터링 후 pandas로 분석 
#             - 조건: 연봉 상위 50% (df['연봉'].median() ) 만 가져오기  후 직급별 평균 연봉 출력
#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성

import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import csv

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8'
}

try:
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
    sql = """
        select jikwonno, jikwonname, busername, jikwonpay, jikwonjik
        from jikwon inner join buser on jikwon.busernum=buser.buserno
    """
    cursor.execute(sql)


    df = pd.DataFrame(cursor.fetchall(),
                       columns=['사번', '이름','부서명', '연봉', '직급'])
    print(df.head(3))
    print()

#      - DataFrame의 자료를 파일로 저장
    with open('jikwoninfo.csv', mode='w', encoding='utf-8') as fobj:
        writer = csv.writer(fobj)
        writer.writerow(df.columns)
        writer.writerows(df.values)

    df2 = pd.read_csv('jikwoninfo.csv')
    print(df2.head(3))
    print()

#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    result = pd.pivot_table(df2, index='부서명', values='연봉', aggfunc=['sum', 'max', 'min'])
    result.columns=['연봉합', '최대', '최소']
    print(result)
    print()

#      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    ctab = pd.crosstab(df['부서명'], df['직급'], margins=True)
    print('교차표\n', ctab)
    print()
    
#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
    sql = """select jikwonno, jikwonname, gogekno, gogekname, gogektel
        from jikwon left outer join gogek on jikwon.jikwonno=gogek.gogekdamsano
    """
    df3 = pd.read_sql(sql, conn)
    df3 = df3.fillna("담당 고객 X")
    print(df3)
    print()

#      - 연봉 상위 20% 직원 출력  : quantile()
    threshold = df2['연봉'].quantile(0.8)
    print(df2[df2['연봉']>=threshold])
    print()

#      - SQL로 1차 필터링 후 pandas로 분석 
#             - 조건: 연봉 상위 50% (df['연봉'].median() ) 만 가져오기  후 직급별 평균 연봉 출력
    sql = "select jikwonjik as 직급, jikwonpay as 연봉 from jikwon"
    df4 = pd.read_sql(sql, conn)
    pay_median = df4['연봉'].median()
    df4 = df4[df4['연봉'] >= pay_median]
    df4_pivot = df4.pivot_table(values='연봉', index='직급', aggfunc='mean')
    print(df4_pivot)
    print()

#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
    buser_ypay = df.groupby(['부서명'])['연봉'].mean()  # 직급별
    print(buser_ypay)
    plt.barh(range(len(buser_ypay)), buser_ypay, alpha=0.4)     # 가로 막대
    plt.yticks(range(len(buser_ypay)), buser_ypay.index)
    plt.xlabel('평균 연봉')
    plt.ylabel('부서별')
    plt.show()

except Exception as e:
    print('처리 오류 : ', e)
    
finally:
    cursor.close()
    conn.close()