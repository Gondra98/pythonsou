#함수 처리

#연습문제1) 리스트를 통해 직원 자료를 입력받아 가공 후 출력하기
# 입력 함수 :  [사번, 이름, 기본급, 입사년도]
def inputfunc():
    datas = [
        [1, "강나루", 1500000, 2010],
        [2, "이바다", 2200000, 2018],
        [3, "박하늘", 3200000, 2005],
    ]
    return datas

datas = inputfunc()

# 처리함수 선언
def processfunc(datas):
    for i in range(len(datas)):
        Basic_Salary = datas[i][2]
        years = 2026 - datas[i][3]                  
        Salary = 0                                  # 급여액 변수선언
        def Salary(Basic_Salary, years):            # 급여액 함수선언 (기본급, 근속년수)
            global Salary
            if years >= 9:
                Salary = Basic_Salary + 1000000     # 근속년수 9년 이상
                datas[i].append(1000000)            # 근속수당 datas리스트 추가
            elif years >= 4:
                Salary = Basic_Salary + 450000      # 근속년수 4년 이상
                datas[i].append(450000)
            else:
                Salary = Basic_Salary + 150000      # 근속년수 0년 이상
                datas[i].append(150000)
            def Net_Salary(Salary):                 # 수령액 함수선언 (급여액, 공제율)
                deductible = 0                      # 공제액 변수선언
                if Salary >= 3000000:               # 급여액 300만원 이상일 경우
                    deductible = Salary * 0.5
                elif Salary >= 2000000:             # 급여액 200만원 이상일 경우
                    deductible = Salary * 0.3
                else:
                    deductible = Salary * 0.15      # 급여액 200만원 미만일 경우
                datas[i].append(int(deductible))
                datas[i].append(int(Salary-deductible))
            Net_Salary(Salary)
        Salary(Basic_Salary,years)

    # 출력
    print('사번\t이름\t기본급\t근무년수\t근속수당\t공제액\t수령액')
    print('------------------------------------------------------------------------------')
    for i in range(len(datas)):
        for j in range(len(datas[i])):
            if j == (len(datas[i])-1):
                print(datas[i][j])
            elif j == 4:
                print(f'\t{datas[i][j]}\t\t',end='')
            else:
                print(f'{datas[i][j]}\t',end='')

processfunc(datas)

print('\n\n\n\n')

# 연습문제2) 리스트를 통해 상품 자료를 입력받아 가공 후 출력하기

# 입력 함수
def inputfunc():
    datas = [
        "새우깡,15",
        "감자깡,20",
        "양파깡,10",
        "새우깡,30",
        "감자깡,25",
        "양파깡,40",
        "새우깡,40",
        "감자깡,10",
        "양파깡,35",
        "새우깡,50",
        "감자깡,60",
        "양파깡,20",
    ]
    return datas

datas = inputfunc()

def processfunc_2(datas):
    for i in range(len(datas)):
        datas[i] = datas[i].split(',')  # 각 리스트 안에 있는 , 기준으로 아이템 쪼갬
        datas[i][1] = int(datas[i][1])  # 수량 정수전환


    # 수량 확인
    for i in range(len(datas)):
        if datas[i][0] == '새우깡':
            datas[i].append(450)
        elif datas[i][0] == '감자깡':
            datas[i].append(300)
        elif datas[i][0] == '양파깡':
            datas[i].append(350)

    # 금액 계산
    for i in range(len(datas)):
        price = datas[i][1] * datas[i][2]
        datas[i].append(price)


    subtotal_num = [0, 0, 0]            # 소계수량 리스트 선언
    subtotal_price = [0, 0, 0]          # 소계가격 리스트 선언

    # 소계수량 계산
    for i in range(len(datas)):
        if datas[i][0] == '새우깡':
            subtotal_num[0] += datas[i][1]
        elif datas[i][0] == '감자깡':
            subtotal_num[1] += datas[i][1]
        elif datas[i][0] == '양파깡':
            subtotal_num[2] += datas[i][1]

    # 소계가격 계산
    for i in range(len(datas)):
        if datas[i][0] == '새우깡':
            subtotal_price[0] += datas[i][3]
        elif datas[i][0] == '감자깡':
            subtotal_price[1] += datas[i][3]
        elif datas[i][0] == '양파깡':
            subtotal_price[2] += datas[i][3]

    total_num = subtotal_num[0] + subtotal_num[1] + subtotal_num[2]             # 총 건수
    total_price = subtotal_price[0] + subtotal_price[1] + subtotal_price[2]     # 총액

    

    
    #출력
    print('상품명\t수량\t단가\t금액')
    print('----------------------------------------------')
    
    for i in range(len(datas)):
        for j in range(len(datas[i])):
            if j == len(datas[i])-1:
                print(datas[i][j])
            else:
                print(f'{datas[i][j]}\t ', end=(''))

    print('\n\n')
    print('소계')
    print(f'새우깡 : {subtotal_num[0]}건   소계액 : {subtotal_price[0]}원')
    print(f'감자깡 : {subtotal_num[1]}건   소계액 : {subtotal_price[1]}원')
    print(f'양파깡 : {subtotal_num[2]}건   소계액 : {subtotal_price[2]}원')

    print('총계')
    print(f'총 건수 : {total_num}')
    print(f'총액 : {total_price}원')

processfunc_2(datas)

