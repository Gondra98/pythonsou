kor = 100   # 모듈의 전역변수

def abc():
    kor = 0     # 함수내의 지역변수
    print('모듈의 멤버 함수')

class My:
    kor = 80        # My 멤버 변수(필드)
    def abc(self):
        print('My 멤버 메소드')

    def show(self):
        #kor = 77    # 메소드 내의 지역변수
        print(kor)
        print(self.kor)
        self.abc()
        abc()

my = My()
my.show()
print('-----------')
print(My.kor)
tom = My()
print(tom.kor)
tom.kor = 88
print(tom.kor)

oscar = My()
print(oscar.kor)


