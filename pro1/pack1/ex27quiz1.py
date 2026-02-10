# 클래스의 상속관계 연습문제 - 다형성 

class ElecProdut:
    volume = 0
    def volumeControl(volume):
        pass

class ElecTv(ElecProdut):
    def volumeControl(self, volume):
        print('티비: ', volume)         # ElecProduct 오버라이딩

class ElecRadio(ElecProdut):
    def volumeControl(self, volume):
        print('라디오: ', volume)       # ElecProduct 오버라이딩


product = ElecProdut()
tv = ElecTv()
product = tv
product.volumeControl(int(input('Tv볼륨 : ')))
print()
radio = ElecRadio()
product = radio
product.volumeControl(int(input('Radio볼륨 : ')))




