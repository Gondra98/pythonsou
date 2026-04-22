# Queue : FIFO 구조
from collections import deque   # list 대신 deque를 Queue 구현
# deque 주요 메소드
# deque(), append(1) : 우측에 추가, appendleft(1) : 좌측에 추가
# pop() : 우측 제거, popleft() : 좌측 제거 ...

# 놀이 공원 대기 줄
queue = deque()
print('놀이 공원 기구 대기 시작')

# 줄서기
queue.append('철수')
print('첫번째 줄서기 : ', list(queue))
queue.append('영희')
print('두번째 줄서기 : ', list(queue))
queue.append('민수')
print('세번째 줄서기 : ', list(queue))

print()
# 놀이 기구 탑승 - FIFO
first_person = queue.popleft()  # queue에서 제거
print(first_person, '놀이 기구 탑승')
print('현재 대기줄 : ', list(queue))
print()

# 한명 더 놀이 기구 탑승 - FIFO ( 중간 데이터 처리 불가 )
first_person = queue.popleft()  # queue에서 제거
print(first_person, '놀이 기구 탑승')
print('현재 대기줄 : ', list(queue))
print()

if queue:
    print('탑승 예정자:', queue[0])
else:
    print('대기자 없음')


print('\n-----------------------------------------------')
# FIFO를 class로 연습
class Queue:
    def __init__(self, iterable = None):
        self._data = deque()
        if iterable is not None:
            for x in iterable:
                self.enqueue(x)

    def enqueue(self, x):
        self._data.append(x)    # back에 요소 추가
        return x
    
    def dequeue(self):          # 앞(front) 요소 제거
        if not self._data:
            raise IndexError('큐 비어 있음')
        return self._data.popleft()
    
    def front(self):    # 큐에서 맨앞 요소를 확인하는 메소드. 조회만 함
        if not self._data:
            raise IndexError('큐 비어 있음')
        return self._data[0]
    
    def is_empty(self):
        return not self._data   # 비었을 때 True 반환
    
    def size(self):
        return len(self._data)
    
    def clear(self):
        self._data.clear()

    def __repr__(self):     # front -> back 순서로 출력하는 특별 메소드
        return f'Queue(front -> back {list(self._data)})'
    

def demo1Func():
    imsi1 = Queue()
    imsi2 = Queue([10, 20, 30])
    print(imsi1)
    print(imsi2)
    print(imsi2.front())
    print(imsi2.size())
    imsi2.clear()
    print(imsi2)
    print('--------------')
    q = Queue()
    for item in ['A', 'B', 'C', 'D']:
        q.enqueue(item)
        print(f'enqueue {item} -> ', q)

    print('FIFO에 따라 하나씩 추출')
    while not q.is_empty():
        print(f'dequeue -> ', q.dequeue(), '|low:', q)
    
def demo2Func(jobs, ppm=15):
    q = Queue(jobs) # 작업들 큐에 입력
    t_sec = 0.0     # 시뮬레이션 시간 누적
    order = []      # 실제 처리된 문서 저장

    print('프린터로 출력하기')
    while not q.is_empty():
        doc, pages = q.dequeue()
        # 출력시간 계산 : 페이지수 / 분당페이지수 * 60
        duration = (pages / ppm) * 60.0
        t_sec += duration
        order.append(doc)   # 처리 순서 기록
        print(f't={t_sec:6.1f}초 | 출력 : {doc:10s}({pages}페이지)')
    
    print('처리순서(FIFO) : ', order)



if __name__ == '__main__':
    demo1Func()
    print('문서 프린터로 출력 시뮬레이션 - FIFO')
    jobs = [('abc.pdf', 10), ('nice.doc', 30), ('good.txt', 5),]
    demo2Func(jobs, ppm=20)     # 현재 프린터는 1분에 20장 출력