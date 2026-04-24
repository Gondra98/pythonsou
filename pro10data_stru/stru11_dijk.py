# Dijkstra 알고리즘
# 가중치가 있는 Graph에서 한 시작점으로 부터 모든 정점까지의 최단 거리를 구하는 알고리즘
# 조건 : 간선 가중치는 음수가 없어야 함. 그래프에는 여러 경로가 있을 수 있다.
# 우선순위큐 (Min Heap)를 사용함
# 작동 방식
# 1) 출발 노드를 설정한다.
# 2) 출발 노드를 기준으로 각 노드의 최소비용을 저장한다.
# 3) 방문하지 않은 노드 중에서 가장 비용이 적은 노드를 선택한다.
# 4) 해당 노드를 거쳐서 특정한 노드로 가는 경우를 고려하여 최소 비용을 갱신한다.
# 5) 더 이상의 최소 노드가 없을 때 까지 위 과정에서 3번, 4번 반복한다.

# 실습 : 노드 A에서 각 노드까지의 최단 거리는?
import heapq


def dijkstraFunc(graph, start):
    dist = {node:float('inf') for node in graph}    # 모든 노드까지의 최단 거리를 무한대로 초기화
    dist[start] = 0     # 'A' = 0   확정된 최단거리만 기억
    print(dist)         # {'A': 0, 'B': inf, 'C': inf, 'D': inf}

    pq = [(0, start)]  # (현재까지의 거리, 노드) : 후보 목록 (갈 수 있는 모든 길)

    while pq:   # 처리할 노드가 남아 있는 동안
        cur_dist, u = heapq.heappop(pq)     # Min Heap (Greedy한 선택)

        if cur_dist > dist[u]:  # 더 짧은 경로가 있으면 현재 노드는 무시(중복 방지)
            continue

        for v, w in graph[u]:   # 현재노드u와 연결된 이웃 노드 v 탐색
            new_dist = cur_dist + w     # u를 거쳐 v로 가는 새 거리 계산
            if new_dist < dist[v]:
                dist[v] = new_dist      # 최단 거리 저장소 (최단 거리 갱신)
                heapq.heappush(pq, (new_dist, v))   # 최단 거리로 큐(Min Heap)에 삽입
    
    return dist


graph = {   # 무방향 형태
    'A':[('B',1),('C',4)],  # A에서 B,C로 가는 가중치 간선
    'B':[('A',1),('C',1),('D',2)],
    'C':[('A',1),('B',1),('D',3)],
    'D':[('B',2),('C',3)]
}


print(dijkstraFunc(graph, 'A')) # {'A': 0, 'B': 1, 'C': 2, 'D': 3}