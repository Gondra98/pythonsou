# 문3) 성별 직원 현황 출력 : 성별(남/여) 단위로 직원 수와 평균 급여 출력  
# 성별 직원수 평균급여  
# 남 3 8500  
# 여 2 7800  

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


        sql = """      
            select jikwongen as 성별, count(*) as 인원수,  
            round(avg(jikwonpay)) as 평균급여  
            from jikwon  
            group by jikwongen  
        """                        

        cursor.execute(sql)  

        datas = cursor.fetchall()  


        for jikwongen, inwonsu, jikwonpay in datas:  
            print(jikwongen, inwonsu, jikwonpay)  



    except Exception as e:  
        print('err : ', e)  

    finally:  
        cursor.close()  
        conn.close()  


if __name__ == "__main__":  
    chulbal()