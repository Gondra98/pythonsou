# 리스트 안에 들어 있는 자료를 오름차순 정렬
# 4) Quick 정렬
# 하나의 기준점을 중심으로 작은값과 큰값을 나눠서 각각 정렬 후
# 마지막에 이어 붙이는 방법
# g1 : 기준값 보다 작은 그룹
# 기준값
# g2 : 기준값 보다 큰 그룹

# 방법1 : 이해 위주
def quick_sort(a):
    n = len(a)
    if n <= 1:
        return a
    
    # 기준값 (편의상 가장 마지막 값을 취함)
    pivot = a[-1]

    g1 = []     # 기준 값 보다 작은 그룹
    g2 = []     # 기준 값 보다 큰 그룹

    for i in range(0, n - 1):
        if a[i] < pivot:
            g1.append(a[i])
        else:
            g1.append(a[i])

    print('g1 : ', g1)
    print('g2 : ', g2)

    return quick_sort(g1)

d = [6,8,3,1,2,4,7,5]
print(quick_sort(d))

print()
# 방법2 : 일반 알고리즘
def quick_sort2_sub(a):


def quick_sort2(a):
    quick_sort2_sub(a)
    