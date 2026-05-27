INF = int(1e9)  # 무한을 의미하는 10억 설정

# 노드의 개수 및 간선의 개수 입력 받기
n, m = map(int, input().split())

# 2차원 리스트를 만들고, 무한으로 초기화
graph = [[INF] * (n + 1) for _ in range(n + 1)]

# 자기 자신에서 자기 자신으로 가는 비용은 0으로 초기화
for a in range(1, n + 1):
    for b in range(1, n + 1):
        if a == b:
            graph[a][b] = 0

# 각 간선에 대한 정보 입력받아, 그 값으로 초기화
for _ in range(m):
    # a에서 b로 가는 비용은 c
    a, b, c = map(int, input().split())
    graph[a][b] = c

# 점화식에 따라 플로이드-워셜 수행
for k in range(1, n + 1):      # 경유 노드
    for a in range(1, n + 1):  # 출발 노드
        for b in range(1, n + 1):  # 도착 노드
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

# 수행된 결과 출력
for a in range(1, n + 1):
    for b in range(1, n + 1):
        # 도달할 수 없는 경우, 무한으로 출력
        if graph[a][b] == INF:
            print(a, "->", b, ": INFINITY", end=' ')
        # 도달할 수 있는 경우 거리를 출력
        else:
            print(a, "->", b, ":", graph[a][b], end=' ')
    print()