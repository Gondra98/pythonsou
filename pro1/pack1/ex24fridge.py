# 클래스의 포함 관계 연습 - 냉장고 객체에 음식 객체 담기

class Fridge:
    isOpened = False
    foods = []

    def open(self):
        self.isOpened = True
        print("냉장고 문을 열기")

    def close(self):
        self.isOpened = False
        print("냉장고 문을 닫기")

    def foodsList(self):        # 냉장고 문이 열린경우 음식물 확인
        for f in self.foods:
            print(f' - {f.name} {f.expiry_date}')
        print()

    def put(self, thing):
        if self.isOpened:
            self.foods.append(thing)
            print(f'냉장고에 {thing.name} 넣음')
            self.foodsList()
        else:
            print('냉장고 문이 닫혀있음')



class FoodData:
    def __init__(self, name, expiry_date):
        self.name = name
        self.expiry_date = expiry_date


fObj = Fridge()

apple = FoodData('사과', '2026-8-1')
fObj.put(apple)
fObj.open()
fObj.put(apple)
fObj.close()

print()
cola = FoodData('콜라', '2027-11-1')
fObj.open()
fObj.put(cola)
fObj.close()

