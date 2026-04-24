# pdf 내용 코드를 구현
import heapq
INF = int(1e9)

# 그래프 (인접 리스트 방식) - (노드번호, 비용) 형태로 저장
graph = [   # 노드 1은 0번째, 노드 2는 1번째 ...
    [(1, 2), (2, 5), (3, 1)],   # 1번 노드 (2번,2), (3번,5), (4번,1)
    [(0, 2), (2, 3), (3, 2)],   # 2번 노드
    [(0, 5), (1, 3), (3, 3), (4, 1), (5, 5)],
    [(0, 1), (1, 2), (2, 3), (4, 1)],
    [(2, 1), (3, 1), (5, 2)],
    [(2, 5), (4, 1)]
]

n = 6   # 노드 갯수
distance = [INF] * n    # 최단 거리 배열 초기화

def dijkstraFunc(start):
    pq = []     # 우선순위큐(Heap)
    distance[start] = 0
    heapq.heappush(pq, (0, start))  # (거리, 노드) 형태로 큐에 삽입

    while pq:
        dist, now = heapq.heappop(pq)

        if distance[now] < dist:
            continue
        
        # 현재 노드에서 갈 수 있는 모든 노드 탐색
        for next_node, cost in graph[now]:
            new_cost = dist + cost

            # 만약 새로운 경로가 더 짧다면
            if new_cost < distance[next_node]:
                distance[next_node] = new_cost
                heapq.heappush(pq, (new_cost, next_node))


dijkstraFunc(0)     # 1번 노드에서 출발

# 각 노드까지의 최단 거리 출력
for i in range(n):
    print(f'{i + 1}번 노드까지 최단거리:{distance[i]}')



