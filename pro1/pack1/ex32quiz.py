# t.data_print()
# r.data_print()  
# s.data_print()

from abc import *

class Employee(metaclass=ABCMeta):
    
    def __init__(self, irum, nai):
        self.irum = irum
        self.nai = nai
    
    @abstractmethod
    def pay(self):      # 추상메소드
        pass

    @abstractmethod
    def data_print(self):       # 추상메소드
        pass

    def irumnai_print(self):    # 이름, 나이 출력용
        print(f'이름: {self.irum}, 나이: {self.nai},', end = ' ')


class Temporary(Employee):                          # Employee 상속
    def __init__(self, irum, nai, ilsu, ildang):    # 일수, 일당 입력
        super().__init__(irum, nai)
        self.ilsu = ilsu
        self.ildang = ildang

    def pay(self):                                  # 추상메소드 Pay 오버라이드 -> 월급 계산 및 출력
        print('월급:', self.ilsu * self.ildang)

    def data_print(self):                           # 추상메소드 data출력
        super().irumnai_print()                        # 상위 클래스 irumnai_print() 호출
        self.pay()


class Regular(Employee):
    def __init__(self, irum, nai, salary):          # 급여 입력
        super().__init__(irum, nai)
        self.salary = salary
    
    def pay(self):                                  # 이미 급여가 있어서 계산 생략
        pass

    def data_print(self):
        super().irumnai_print()                        # 상위 클래스 irumnai_print() 호출
        print('급여:', self.salary)


class Salesman(Regular):                                        # Regular 메소드 상속
    def __init__(self, irum, nai, salary, sales, comission):    # 급여,실적,수수료율 입력
        super().__init__(irum, nai, salary)
        self.sales = sales
        self.comission = comission
    
    def pay(self):                                              # 수령액 계산
        allowance = self.salary + (self.sales * self.comission)
        allowance = int(allowance)                              # 계산한 수령액 정수화
        return allowance
    
    def data_print(self):
        super().irumnai_print()                                    # 상위 클래스 irumnai_print() 호출
        print('수령액:', self.pay())






t = Temporary('홍길동', 25, 20, 15000) 
t.data_print()                                  # 이름: 홍길동, 나이: 25, 월급: 300000



r = Regular('한국인', 27, 3500000)
r.data_print()                                  # 이름: 한국인, 나이: 27, 급여: 3500000


s = Salesman('손오공', 29, 1200000, 5000000, 0.25)
s.data_print()                                  # 이름: 손오공, 나이: 29, 수령액: 2450000




