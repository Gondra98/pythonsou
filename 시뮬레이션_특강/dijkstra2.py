import heapq

INF = int(1e9)  # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수
n, m = map(int, input().split())

# 각 노드에 연결되어 있는 노드에 대한 정보 담는 리스트 만들기
graph = [[] for i in range(n + 1)]

# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n + 1)

for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))
    graph[b].append((a, c))  # 양방향 간선

def dijkstra(start):
    q = []
    heapq.heappush(q, (0, start))
    distance[start] = 0
    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
            continue
        for i in graph[now]:
            cost = dist + i[1]
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

dijkstra(1)
print(distance[-1])  # N번 노드(작전목표지점)까지 최소 적 전차 수