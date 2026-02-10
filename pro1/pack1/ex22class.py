# 클래스는 새로운 타입을 만들어 자원을 공유 가능

# class Singer:
#     title_song = "빛나라 대한민국"

#     def sing(self):
#         msg = "노래는 "
#         print(msg, self.title_song)

#import ex22singer
from ex22singer import Singer

#bts = ex22singer.Singer()  # 생성자 호출해 객체를 생성 후 주소를 bts에 치환
bts = Singer()
bts.sing()
print(type(bts))
bts.title_song = "Permission to Dance"
bts.sing()
bts.co = '빅히트 엔터테인먼트'
print('소속사 : ', bts.co)


print()
ive = Singer()
ive.sing()
print(type(ive))
# print('소속사 : ', ive.co)  # AttributeError: 'Singer' object has no attribute 'co'

Singer.title_song = "아 대한민국"
bts.sing()          # bts 객체에 있는 변수참조
ive.sing()          # Singer클래스에 있는 변수 참조

niceGroup = ive     # ive주소 치환
niceGroup.sing()
