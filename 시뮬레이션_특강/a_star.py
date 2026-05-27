class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def heuristic(node, goal, D=1, D2=2 ** 0.5):
    # Diagonal Distance
    dx = abs(node.position[0] - goal.position[0])
    dy = abs(node.position[1] - goal.position[1])
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def aStar(maze, start, end):
    startNode = Node(None, start)
    endNode = Node(None, end)

    openList = []
    closedList = []
    openList.append(startNode)

    while openList:
        currentNode = openList[0]
        currentIdx = 0

        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index

        openList.pop(currentIdx)
        closedList.append(currentNode)

        # 목적지 도달 시 경로 반환
        if currentNode == endNode:
            path = []
            current = currentNode
            while current is not None:
                # x, y = current.position
                # maze[x][y] = 7  # 경로 표시하려면 주석 해제
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []

        # 8방향 탐색
        for newPosition in [(0, -1), (0, 1), (-1, 0), (1, 0),
                            (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nodePosition = (
                currentNode.position[0] + newPosition[0],
                currentNode.position[1] + newPosition[1])

            within_range_criteria = [
                nodePosition[0] > (len(maze) - 1),
                nodePosition[0] < 0,
                nodePosition[1] > (len(maze[len(maze) - 1]) - 1),
                nodePosition[1] < 0,
            ]
            if any(within_range_criteria):
                continue

            if maze[nodePosition[0]][nodePosition[1]] != 0:
                continue

            new_node = Node(currentNode, nodePosition)
            children.append(new_node)

        # 자식 노드 처리
        for child in children:
            # closedList에 있으면 skip
            if child in closedList:
                continue

            # g, h, f 계산
            child.g = currentNode.g + 1
            child.h = ((child.position[0] - endNode.position[0]) ** 2) + \
                      ((child.position[1] - endNode.position[1]) ** 2)
            # child.h = heuristic(child, endNode)  # 다른 휴리스틱 사용 시
            # print("position:", child.position)       # 거리 추정값 확인
            # print("from child to goal:", child.h)
            child.f = child.g + child.h

            # openList에 있고 g값이 더 크면 skip
            if len([openNode for openNode in openList
                    if child == openNode and child.g > openNode.g]) > 0:
                continue

            openList.append(child)


def main():

    # 1은 장애물
    maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (7, 6)

    path = aStar(maze, start, end)
    for y,x in path:
        maze[y][x] = 8
    for i in maze:
        print(i)

    print()
    print(path)


if __name__ == '__main__':
    main()