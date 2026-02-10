# 커피 자판기 프로그램
# 조건
# 입력자료는 키보드를 사용
# 커피는 한잔에 200원.
# 100원 넣고 커피를 요구하면 요금 부족 메시지 출력
# 400원 넣고 2잔 요구하면 두 잔 출력
# 500원 넣고 1잔 요구하면 300원 반납

class CoinIN:
    def __init__(self, coin):
        self.coin = coin
    
    def culc(self, cupCount):
        self.cupCount = cupCount        # 해당 객체 컵카운트 멤버 생성
        price = 200 * self.cupCount     # 총가격      
        quotient = (price // self.coin)   # 총가격 // 동전 => 몫
        remainder = price % self.coin       # 동전으로 총가격 나눌때 나머지

        if price < self.coin:                   # 동전이 커피 총가격보다 많은경우
            self.change = self.coin - price     # 잔돈 계산
        elif price > self.coin:                 # 총가격이 동전보다 클경우
            if remainder == 0:                  # 요금부족
                print('요금 부족')
                self.cupCount = 0
                self.change = self.coin
            elif remainder > 0:                 # 어느정도만 살수있는경우
                self.change = self.coin - price 
                self.cupCount = quotient        # 컵은 몫만큼 나옴
        elif price == self.coin:
            self.change = 0



class Machine:
    cupCount = 1

    def showData(self):
        print('출력형태--------------')
        coffee = CoinIN(int(input('동전을 입력하세요 : ')))                 # CoinIN 클래스 포함
        self.cupCount = coffee.culc(int(input('몇 잔을 원하세요 : ')))
        print(f'커피 {coffee.cupCount}잔과 잔돈 {coffee.change}원')


print(__name__)

if __name__ == '__main__':
    machine = Machine()
    machine.showData()

