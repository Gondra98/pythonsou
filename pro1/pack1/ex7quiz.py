# Python 기초 연습문제
# https://cafe.daum.net/flowlife/RUrO/40

print('1번문제 정답\n')

sum = 0 
for i in range(1, 101):
    if i % 3 == 0 and i % 2 != 0:
        sum += i 

print(f'{sum}\n')

# 다른풀이
# result = sum([i for i in range(1, 101) if i % 3 == 0 and i % 2 != 0])
# print(result)

print('2번문제 정답\n')

for i in range(2, 6):
    for j in range(1, 6):
        print(f'{i}*{j}={i*j}', end = ' ')
    print()
print()

print('3번문제 정답')
sum3 = 0 
for i in range(1, 101):
    if i % 2 == 0:
        sum3 += i
    else: sum3 -= i 

print(f'{sum3}\n')

print('4번문제 정답\n')

sum4 = 0
for i in range(1, 100, 2):
    i *= -1
    sum += i
print(sum)

print('5번문제 정답\n')

list5 = []
for i in range(101):
    a = i / 10
    b = i % 10
    if a + b >= 10:
        list5.append(i)

print(list5,'\n')

############## 다른 방식 ##############
# list5 = [i for i in range(101) if i % 10 + i/10 >=10]
# print(list5,'\n')
#######################################

print('6번문제 정답\n')

sum6 = 0 
count = 1
while True:
    sum6 += count
    if sum6 > 1000:
        break
    count += 1

print(f'{count}, {sum6}\n')

print('7번문제 정답\n')
for i in range(2, 10):
    for j in range(1, 10):
        if i * j  > 30: break
        print(f'{i}*{j}={i*j}', end = ' ')
    print()
print()

print('8번문제 정답\n')
list8 = []
a = 2 
while a <= 1000:
    b = 2
    is_prime = True 
    while b < a:
        if a % b == 0: 
            is_prime = False
            break
        b += 1
    if is_prime:
        list8.append(a)
    a += 1
print(f'리스트: {list8}, 개수:{len(list8)}')

print('정답은'+'입니다')

print('9번문제 정답\n')
# for i in range(51):
#     if i % 3 == 0: continue
#     else: print(f"{i}", end = ' ')
list9 = [i for i in range(51) if i % 3 != 0]
print(list9)

print('10번문제 정답\n')

list10 = []
for i in range(101):
    if i % 4 != 0 and i % 6 != 0:
        list10.append(i)

sum = 0 
tmp = []
for j in list10:
    if j % 5 == 0:
        tmp.append(j)
        sum += j

print(tmp, sum)