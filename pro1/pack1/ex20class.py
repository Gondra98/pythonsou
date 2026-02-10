class Car:                          #공유 가능, 대문자로 시작
    handle = 1
    speed = 0

    def __init__(self, name, speed):
        self.name = name        # 현재객체의name에게 name(지역변수) 인자값 치환
        self.speed = speed      
    
    def showData(self):
        km = "킬로미터"
        msg = "속도:" + str(self.speed) + km
        return msg
    
    def printHandle(self):
        return self.handle     # self를 안적으면 해당 지역변수를 참조 self를 작성하면 해당 객체 변수참조
    
print(Car.handle)       # 원형(prototype) 클래스의 멤버 호출
car1 = Car('tom', 10)   # 생성자 호출 후 객체 생성(인스턴스화)
print('car1 객체 주소:', car1)
print('car1 : ', car1.name, ' ', car1.speed, car1.handle)   # handle은 없으니깐 handle만 원형클래스 참조
car1.color = '파랑'                                         # 멤버추가 가능
print('car1.color : ', car1.color)


car2 = Car('john', 20)  # 생성자 호출 후 객체 생성(인스턴스화)
print('car2 객체 주소:', car2)
print('car2 : ', car2.name, ' ', car2.speed, car2.handle)
# print(Car.color, ' ', car2.color)
print(Car, car1, car2)
print(id(Car), id(car1), id(car2))
print(car1.__dict__)
print(car2.__dict__)


print('---메소드------------------')
print('car1 speed : ', car1.showData())     # 인터프리터가 car1.showData(car1), car1을 직접 넣어줌 그래서 작성 안함 넣으면 오히려 오류발생
print('car2 speed : ', car2.showData())
car1.speed = 80
car2.speed = 110
print('car1 speed : ', car1.showData())
print('car2 speed : ', car2.showData())

print('car1 handle : ', car1.printHandle())
print('car2 handle : ', car2.printHandle())