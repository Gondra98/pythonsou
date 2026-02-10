# 재귀문제 :  리스트 자료 v = [7, 9, 15, 43, 32, 21] 에서 최대값 구하기 - 재귀 호출 사용 




def find_max(data, len_data):
    max = data[len_data-1]          # max를 리스트 마지막 값 초기화


    if len_data-1 == 0:             # 인덱스 0이면 반환
        return max
    
    temp = find_max(data, len_data-1)   # 나머지 부분 최대값 재귀 호출
    

    if max < temp:                      # 현재 값과 재귀값 비교
        max = temp

    

    return max


v = [7, 9, 15, 43, 32, 21]

print(find_max(v, len(v)))