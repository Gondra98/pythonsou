# Binary Tree(이진트리) - 자식이 둘 이하(차수가 2이하)인 노드로 구성된 트리
# 노드 방문 방법 3가지 : pre-order(전위), in-order(중위), post-order(후위)
# 이진트리 순회는 DFS(깊이 우선 탐색) 기반


tree = {
    'A':('B', 'C'),
    'B':('D', 'E'),
    'C':(None, None),
    'D':(None, None),
    'E':(None, None)
}

# 전위순회
def preOrder(node):
    if node is None:
        return
    print(node, end=' ')
    left, right = tree[node]
    preOrder(left)  # 재귀
    preOrder(right)

# 중위순회
def inOrder(node):
    if node is None:
        return
    left, right = tree[node]
    inOrder(left)  # 재귀
    print(node, end=' ')
    inOrder(right)

# 후위순회
def postOrder(node):
    if node is None:
        return
    left, right = tree[node]
    postOrder(left)  # 재귀
    postOrder(right)
    print(node, end=' ')


print('전위 순회 결과 : ')
preOrder('A')
print('중위 순회 결과 : ')
inOrder('A')    # BST(Binary Search Tree) 오름차순 정렬
print('후위 순회 결과 : ')
postOrder('A')

