# 예외처리 : 파일, 네트워크, DB작업, 실행에러 등의 에러 대처

def divide(a, b):
    return a / b

print('이런 저런 작업 진행...')
# c = divide(5, 2)
# c = divide(5, 0)
# print(c)

# try except문법은 할까고민할때 써라 써도 나쁠게 없음
try:
    # 실행문(예외 발생 가능 구문)
    c = divide(5, 2)
    # c = divide(5, 0)
    print(c)

    aa = [1, 2]
    # print(aa[0])
    # print(aa[3])

    open("c:/work/abc.txt")

except ZeroDivisionError:               # 예외 종류 관련 클래스
    print('두번째 값은 0을 주면 안돼요')    # 예외 발생 처리 구문
except IndexError as err:                   # as로 에러용어 생성
    print('참조 범위 오류 : ', err)         # err로 어떤 오류인지 확인함
except Exception as e:                     # 예외 처리 상위클래스 Exception, 순서 주의
    print('에러 : ', e)
finally:
    print('에러 유무에 상관없이 반드시 수행됨')

print('end')
print('종료')