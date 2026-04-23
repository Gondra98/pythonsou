# 데큐(Deque) : 양쪽 끝에서 삽입과 삭제가 모두 가능한 자료규조
# 놀이공원 우선 탑승 + 일반 대기 줄

from collections import deque

dq = deque()
print('놀이공원 대기 시작')

# 일반인은 뒤쪽으로 들어옴(Queue 처럼)
dq.append('철수')
dq.append('영희')
dq.append('민수')
print('일반 대기 : ', list(dq))
print()


# VIP 고객(지수)은 앞쪽으로 들어옴
dq.appendleft('VIP지수')
print('현재 대기 줄 상태: ', list(dq))
print()

# 놀이기구에 탑승
person = dq.popleft()
print(person, '탑승')
print('현재 대기 줄 상태2: ', list(dq))
print()

# 줄 맨 뒷사람 줄서기 포기
person = dq.pop()
print(person, ': 줄서기 포기')
print('현재 대기 줄 상태3: ', list(dq))


# 실시간 로그분석, 이동 평균 계산 등에 활용가능

