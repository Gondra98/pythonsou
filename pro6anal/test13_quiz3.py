# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

import pandas as pd
import numpy as np
from scipy import stats
import MySQLdb

try:
    conn = MySQLdb.connect(host='127.0.0.1', user='root', passwd='123', db='test', charset='utf8')
    sql = """
        SELECT b.busername, j.jikwonpay 
        FROM jikwon j 
        JOIN buser b ON j.busernum = b.buserno 
        WHERE b.busername IN ('총무부', '영업부')
    """
    df = pd.read_sql(sql, conn)
except Exception as e:
    pass
finally:
    conn.close()

# 결측치 처리 
df['jikwonpay'] = df['jikwonpay'].fillna(
    df.groupby('busername')['jikwonpay'].transform('mean')
)

# 데이터 분리
total_dept = df[df['busername'] == '총무부']['jikwonpay']
sales_dept = df[df['busername'] == '영업부']['jikwonpay']


# 귀무 : 총무부와 영업부 직원의 연봉 평균에 차이가 없다.
# 대립 : 총무부와 영업부 직원의 연봉 평균에 차이가 있다.

# 정규성 검정
shapiro_total = stats.shapiro(total_dept)
shapiro_sales = stats.shapiro(sales_dept)
print(f"총무부 p-value: {shapiro_total.pvalue:.4f}")
print(f"영업부 p-value: {shapiro_sales.pvalue:.4f}")
# p < 0.05 정규성 만족 안함

u_stat, p_val = stats.mannwhitneyu(total_dept, sales_dept, alternative='two-sided')
print(f"p-value: {p_val:.4f}")
# 결론: p-value 0.4721 가 0.05보다 크므로 귀무가설을 채택
# 총무부와 영업부 간의 연봉 평균에 통계적으로 유의미한 차이가 있다고 볼 수 없음.