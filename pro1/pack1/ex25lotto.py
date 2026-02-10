import random

class LottoBall:
    def __init__(self, num):
        self.num = num

class LottoMachine:
    def __init__(self):
        self.ballist = []
        for i in range(1, 46):
            self.ballist.append(LottoBall(i))   # 포함

    def selectBalls(self):
        # for a in range(45):
        #     print(self.ballist[a].num, end = ' ')
            
        print('------------')
        random.shuffle(self.ballist)    # 번호 섞기
        # for a in range(45):
        #     print(self.ballist[a].num, end = ' ')
        
        return self.ballist[0:6]

class LottoUI:
    def __init__(self):
        self.machine = LottoMachine()   # 포함 관계

    def playLotto(self):
        input('로또를 시작하려면 엔터키를 누르세요')
        selectedBalls = self.machine.selectBalls()
        for ball in selectedBalls:
            print("%d"%(ball.num))


if __name__ == '__main__':
    # machine = LottoMachine()
    # print(machine.selectBalls())

    # lot = LottoUI()
    # lot.playLotto()
    LottoUI().playLotto()
