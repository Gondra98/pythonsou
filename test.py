def selection_sort_with_counts(arr):
    n = len(arr)
    compare_count = 0
    swap_count = 0
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
                compare_count += 1
                swap_count += 1
            else:
                compare_count += 1
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    print("정렬 결과:", arr)
    print("비교 횟수:", compare_count)
    print("교환 횟수:", swap_count)
    return arr

selection_sort_with_counts([64, 25, 12, 22, 11])