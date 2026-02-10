# function : 여러개의 수행문을 하나의 이름으로 묶은 실행 단위
# 함수 고유의 실행 공간을 갖음
# 자원의 재활용

# 내장함수 일부 체험
print(sum([1,2,3]))
print(bin(8)) # 이진수
print(eval('4+5')) 
print(round(1.2), round(1.6)) # 반올림
import math # math 모듈 호출
print(math.ceil(1.2), ' ', math.ceil(1.2)) # 올림
print(math.floor(1.2), ' ', math.floor(1.2)) # 내림

b_list = [True, 1, False]
print(all(b_list)) # 모든 항목이 참일 때 참 
print(any(b_list)) # 하나라도 항목이 참이면 참

data1 = [10, 20, 30]
data2 = ['a', 'b', 'c']
for i in zip(data1, data2): # 집합을 만들어주는 함수
    print(i)