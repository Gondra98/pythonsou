INF = int(1e9)

def bf(start):
    dis[start] = 0  # 시작 지점 초기화

    # 매 반복마다 모든 간선 확인
    # 음의 간선 사이클 존재 유무가 필요하면 n번과 return 처리
    # 필요 없다면 n-1번과 리턴 처리는 필요 없음, dis 테이블만 필요함
    for i in range(n + 1):
        # 모든 간선 확인
        for j in range(m):
            current = edges[j][0]
            next_node = edges[j][1]
            cost = edges[j][2]
            # 시작위치에서 현재 노드까지 이동이 가능하면서
            # 현재 간선을 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if dis[current] != INF and dis[next_node] > cost + dis[current]:
                dis[next_node] = dis[current] + cost
                # 사이클 유무 확인을 위해 n번 돌렸을 때
                # 최단 거리 갱신이 발생하면 음의 사이클이 존재
                if i == n - 1:
                    return True
    return False

# 노드, 간선 개수
n, m = map(int, input().split())
edges = []
dis = [INF] * (n + 1)  # 최단 거리 테이블

# 간선 정보
for _ in range(m):
    a, b, c = map(int, input().split())
    edges.append((a, b, c))

cycle = bf(1)

if cycle:  # 음의 사이클 발생
    print(-1)
else:
    # 1번 노드에서 시작했으니 다른 노드로 가기 위한 최단 거리 출력
    for i in range(2, n + 1):
        if dis[i] == INF:
            print(-1)
        else:
            print(dis[i])