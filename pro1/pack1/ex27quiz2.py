# 다중 상속 연습문제 
# Wolf는 Dog, Cat, Fox는 Cat, Dog

# 동물 상위 클래스
class Animal():

    def __init__(self):
        print('동물 생성')

    def move(self):
        print('움직이는 생물')

# 강아지 클래스
class Dog(Animal):                  # Animal 상속

    def __init__(self, name):
        self.name = name
        print('강아지 생성')

    def move(self):
        print('움직이는 강아지')

# 고양이 클래스
class Cat(Animal):                  # Animal 상속

    def __init__(self, name):
        self.name = name
        print('고양이 생성')

    def move(self):
        print('움직이는 고양이')

# 늑대 클래스                       # Dog, Cat 다중상속
class Wolf(Dog, Cat):
    def __init__(self):
        print('늑대 생성')

# 여우 클래스                       # Cat, Dog 다중상속
class Fox(Cat, Dog):

    def __init__(self):
        print('여우 생성')
    
    def move(self):
        print('움직이는 여우')

    def foxMethod(self):
        print('foxMethod 실행')



ani = Animal()
ani.move()
print()

dog = Dog("멍멍이")
dog.move()
print()

cat = Cat("시리")
cat.move()
print()

wolf = Wolf()
wolf.move()
print()

fox = Fox()
fox.move()
fox.foxMethod()