import sys

v, e = map(int, input().split())

INF = 100000000

# 2차원 리스트 초기화 (자기 자신 포함 전부 무한)
s = [[INF] * v for i in range(v)]

# 간선 정보 입력
for i in range(e):
    a, b, c = map(int, input().split())
    s[a - 1][b - 1] = c  # 0-indexed

# 플로이드-워셜 수행
for k in range(v):
    for i in range(v):
        for j in range(v):
            if s[i][j] > s[i][k] + s[k][j]:
                s[i][j] = s[i][k] + s[k][j]

# 최소 사이클 탐색 (자기 자신으로 돌아오는 비용)
result = INF
for i in range(v):
    result = min(result, s[i][i])

if result == INF:
    print(-1)
else:
    print(result)