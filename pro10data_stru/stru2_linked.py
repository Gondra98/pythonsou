# 연결된 리스트 (Linked List)
# 자료를 임의의 공간에 기억시키고, 순서에 따라 포인터로 자료를 연결시킨 구조

# 놀이공원에 줄서기
class Node:
    def __init__(self, name):
        # 이름과 다음 사람을 가리키는 next를 가장
        self.name = name
        self.next = None    # pointer

# 연결 리스트를 관리
class LinkedList:
    def __init__(self):
        self.head = None    # 맨 앞사람 주소(list의 시작점)
    
    # 새로운 Node를 추가(줄 뒤에 다음 사람 추가)
    def append(self, name):
        new_node = Node(name)   # 새 Node 생성

        if self.head is None:   # 줄(List)이 비어 있는 경우
            self.head = new_node
            return
        
        # 줄의 맨 끝 사람 찾기(이미 노드가 있다면 마지막 노드까지 이동)
        current = self.head
        while current.next:
            current = current.next

        current.next = new_node

    def show(self):
        # print(line.head)
        # print(line.head.next)
        # print(line.head.next.next)
        current = self.head
        while current:
            print(current.name, end="->")
            current = current.next
        print('끝')

    # 특정 사람 뒤에 새사람 끼워 넣기
    # target노드를 찾고 -> 새 노드 만들고 -> 기존 연결 변경
    def insert_after(self, target_name, new_name):
        current = self.head

        while current:
            if current.name == target_name:
                new_node = Node(new_name)   # 지수 객체 생성
                new_node.next = current.next
                current.next = new_node
                return
            current = current.next

    # 특정 사람 삭제
    def remove(self, name):
        # 맨 앞사람이 나가는 경우
        if self.head and self.head.name == name:
            self.head = self.head.next  # head를 두번째 노드의 주소를 변경
            return
        
        # 첫 노드가 삭제 대상이 아닌 경우
        current = self.head
        while current and current.next:
            if current.next.name == name:
                current.next = current.next.next
                return
            current = current.next


line = LinkedList()
line.append('철수')     # 추가
line.append('영희')
line.append('민수')

print('현재 줄 상태 : ')
line.show()
print()

# 영희 뒤에 지수를 삽입
line.insert_after('영희', '지수')
print('지수를 삽입 줄 상태 : ')
line.show()
print()

# 영희가 줄서기를 포기 (삭제)
line.remove('영희')
print('영희 삭제 후 줄 상태 : ')
line.show()
print()

