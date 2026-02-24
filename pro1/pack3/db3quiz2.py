# 문2) 직원번호와 직원명을 입력(로그인)하여 성공하면 아래의 내용 출력
# 직원번호 입력 : _______
# 직원명 입력 : _______
# 직원번호 직원명 부서명 부서전화 직급 성별
#      1 홍길동 총무부 111-1111 이사 남


import MySQLdb


config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8'
}


def chulbal():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()

        jik_no = input('직원번호 입력:')
        # print(jik_no)
        jik_name = input('직원명 입력:')
        # print(jik_name)
        sql = """    
            select jikwonno as 직원번호, jikwonname as 직원명, 
            busername as 부서명, busertel as 부서전화,
            jikwonjik as 직급, jikwongen as 성별
            from jikwon
            inner join buser on busernum= buserno
            where jikwonno={0} and jikwonname='{1}'
        """.format(jik_no, jik_name)                           
        # print(sql)

        cursor.execute(sql)

        datas = cursor.fetchall()
        # print(datas)

        if len(datas) == 0:
            print("로그인 실패")
            return      # sys.exit(0)
        
        for jikwonno, jikwonname, busername, busertel, jikwonjik, jikwongen in datas:
            print(jikwonno, jikwonname, busername, busertel, jikwonjik, jikwongen)



    except Exception as e:
        print('err : ', e)

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    chulbal()



# 문2-1) 직원번호와 직원명을 입력(로그인)하여 성공하면 아래의 내용 출력
# 해당 직원이 근무하는 부서 내의 직원 전부를 직급별 오름차순으로 출력. 직급이 같으면 이름별 오름차순한다.
# 직원번호 입력 : _______
# 직원명 입력 : _______
# 직원번호 직원명 부서명 부서전화 직급 성별
# 1 홍길동 총무부 111-1111 이사 남
# ...
# 직원 수 :
# 이어서 로그인한 해당 직원이 관리하는 고객 자료도 출력한다.
# 고객번호 고객명 고객전화 나이
# 1 사오정 555-5555 34
# 관리 고객 수 :

import MySQLdb


config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'123',
    'database':'test',
    'port':3306,
    'charset':'utf8'
}


def chulbal():
    try:
        conn = MySQLdb.connect(**config)
        cursor = conn.cursor()

        jik_no = input('직원번호 입력:')
        # print(jik_no)
        jik_name = input('직원명 입력:')
        # print(jik_name)
        sql = """    
            select jikwonno as 직원번호, jikwonname as 직원명, 
            busername as 부서명, busertel as 부서전화,
            jikwonjik as 직급, jikwongen as 성별
            from jikwon
            inner join buser on busernum= buserno
            where busernum = (select buserno from jikwon where jikwonno = {0} and jikwonname = '{1}')
            order by jikwonjik, jikwonname asc
        """.format(jik_no, jik_name)                           
        # print(sql)

        cursor.execute(sql)

        datas = cursor.fetchall()
        # print(datas)

        if len(datas) == 0:
            print("로그인 실패")
            return      # sys.exit(0)
        
        for jikwonno, jikwonname, busername, busertel, jikwonjik, jikwongen in datas:
            print(jikwonno, jikwonname, busername, busertel, jikwonjik, jikwongen)



    except Exception as e:
        print('err : ', e)

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    chulbal()