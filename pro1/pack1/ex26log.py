# 첨도, 왜도가 큰 (편차가 큰 데이터 ...) 데이터를 로그변환하면 분포 개선, 범위 차이 축소 등으로 인해
# 분포 개선, 범위 차이 축소 등으로 인해 모델을 안정적으로 수행 가능

import math

class LogTrans:
    def __init__(self, offset: float=1.0):
        self.offset = offset

    def transform(self, x_list:list[float]):    # 로그 변환
        return [math.log(x + self.offset) for x in x_list]
    
    def inverse_trans(self, x_list:list[float]):    # 역 변환
        return [math.exp(x_log) - self.offset for x_log in x_list]

def main():
    data = [10.0, 100.0, 1000.0, 10000.0]   # 예: 편차가 큰 자료들
    # 로그 변환용 객체
    log_trans = LogTrans(offset=1.0)
    
    # 로그 변환 및 역변환
    data_log_scaled = log_trans.transform(data)
    reversed_data = log_trans.inverse_trans(data_log_scaled)
    reversed_data_round = [ round(val, 1) for val in reversed_data ]

    print('원본 자료 : ', data)
    print('로그변환 자료 : ', data_log_scaled)
    print('역변환 자료 : ', reversed_data_round)


if __name__ == "__main__":
    main()