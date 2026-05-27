import sys
input = sys.stdin.readline

INF = int(1e9)

def bellmanFord():
    dist = [0] * (N + 1)  # 모든 노드를 시작점으로 (전부 0)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            for wei, vec in adjList[j]:
                if dist[vec] > wei + dist[j]:
                    dist[vec] = wei + dist[j]
                    if i == N:  # N번째 순회에서도 갱신되면 음수 사이클
                        return True  # 음수 사이클 존재
    return False

TC = int(input())
for _ in range(TC):
    N, M, W = map(int, input().split())
    adjList = [[] for _ in range(N + 1)]

    for _ in range(M):
        S, E, cost = map(int, input().split())  # T → cost로 변경
        adjList[S].append((cost, E))
        adjList[E].append((cost, S))

    for _ in range(W):
        S, E, cost = map(int, input().split())
        adjList[S].append((-cost, E))          # 웜홀: 단방향 음수 간선

    print("YES" if bellmanFord() else "NO")