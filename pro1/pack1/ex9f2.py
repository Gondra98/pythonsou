# 사용자 정의 함수
'''
def 함수명(가인수,,,,):
    return 반환값           #하나의 값만 반환. return이 없을 경우 None을 반환함.

    
함수명(실인수,,,,)          # 호출방법
'''

def doFunc1():
    print('doFunc1 수행')
    return None

def doFunc2(name):
    print('name : ',name)
    return None

def doFunc3(arg1, arg2):
    re = arg1 + arg2
    return re

def doFunc4(a1, a2):
    imsi = a1 + a2
    if imsi % 2 == 0:
        return None
    elif imsi % 2 == 1:
        return imsi

doFunc1()
print('함수 주소는 ',doFunc1)
print('함수 주소(해시태그)는 ',id(doFunc1))

doFunc1()
doFunc2('길동')
print(doFunc3(20, 80))
print(doFunc3('20','80'))
print(doFunc3('대한','민국'))

def triArea(a,h):
    Area = a * h / 2
    triAreaPrint(Area)
def triAreaPrint(Aa):
    print('삼각형 면적은 ', Aa)

triArea(20,30)

def passResult(kor, eng):
    ss = kor + eng
    if ss >= 50:
        return True
    else:
        return False
if passResult(40,20):
    print('합격')
else:
    print('불합격')

def swapFunc(a,b):
    return b, a
a=10; b=20
print(a,' ',b)
print(swapFunc(a,b))

def funcTest():
    print('funxcTest 멤버 처리')
    def funcInner():
        print('내부 함수')
    funcInner()

funcTest()

def isOdd(para):
    return para%2==1
mydict={x:x*x for x in range(11) if isOdd(x)}
print(mydict)

print('변수의 생존범위 (scope rule)----')



print()
a = 10; b = 20; c = 30  # 모듈 단위
def Foo():
    a = 7       # 지역 변수'
    b = 100
    def Bar():
        global c    # Bar 함수의 멤버 아니라 모듈(파일)의 멤버가 됨. 전역변수
        nonlocal b
        b = 8       # 지역 변수
        print(f'함수 수행 후 a:{a}, b:{b}, c:{c}')
        c = 9
        b = 200
    Bar()
    print(f'함수 수행 후 a:{a}, b:{b}, c:{c}')

Foo()
print(f'함수 수행 후 a:{a}, b:{b}, c:{c}')

print()
g = 1
print('g : ', g)
def func():
    global g
    a = g       # 지역변수를 선언하면 지역변수를 먼저 참조
    g = 2
    return a

print(func())
print('g : ', g)