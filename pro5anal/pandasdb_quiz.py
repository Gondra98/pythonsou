# c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.
#       조건 :  try ~ except MySQLdb.OperationalError as e:      사용
#      사번  직원명  부서명   직급  부서전화  성별
#      ...
#      인원수 : * 명
#     - 성별 연봉 분포 + 이상치 확인    <== 그래프 출력
#     - Histogram (분포 비교) : 남/여 연봉 분포 비교    <== 그래프 출력


import pymysql
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123',
    'database': 'test',
    'port': 3306,
    'charset': 'utf8'
}


jikwon_no   = input("사번  입력 : ").strip()
jikwon_name = input("직원명 입력 : ").strip()


try:
    conn   = pymysql.connect(**config)
    cursor = conn.cursor()

    sql_login = """
        SELECT j.jikwonno, j.jikwonname, b.busername,
               j.jikwonjik, b.busertel, j.jikwongen
        FROM   jikwon j
        INNER JOIN buser b ON j.busernum = b.buserno
        WHERE  j.jikwonno = %s
          AND  j.jikwonname = %s
    """
    cursor.execute(sql_login, (jikwon_no, jikwon_name))
    login_row = cursor.fetchone()

    if login_row is None:
        print("로그인 실패 : 사번 또는 직원명이 일치하지 않습니다.")

    else:
        print("로그인 성공\n")

        sql_all = """
            SELECT j.jikwonno   AS 사번,
                   j.jikwonname AS 직원명,
                   b.busername  AS 부서명,
                   j.jikwonjik  AS 직급,
                   b.busertel   AS 부서전화,
                   j.jikwongen  AS 성별
            FROM   jikwon j
            INNER JOIN buser b ON j.busernum = b.buserno
        """
        df = pd.read_sql(sql_all, conn)

        print(df.to_string(index=False))
        print(f"\n인원수 : {len(df)} 명")


        # ── 그래프1 : 성별 연봉 분포 + 이상치 (Boxplot) ──────
        axes[0].boxplot([male, female], labels=['남', '여'])
        axes[0].set_title('성별 연봉 분포 + 이상치 확인')
        axes[0].set_ylabel('연봉')

except pymysql.OperationalError as e:
    print('DB 오류 :', e)
except Exception as e:
    print('처리 오류 :', e)
finally:
    try:
        cursor.close()
        conn.close()
    except:
        pass