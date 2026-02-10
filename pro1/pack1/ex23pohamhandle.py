# 어딘가에서 필요한 부품 핸들 클래스 작성
class PohamHandle:
    quantity = 0    # 핸들 회전량 ,멤버필드, 공유자원

    def leftTurn(self, quantity):       # 여기 quantity는 지역변수
        self.quantity = quantity
        return "좌회전"
    
    def rightTurn(self, quantity):
        self.quantity = quantity
        return "우회전"