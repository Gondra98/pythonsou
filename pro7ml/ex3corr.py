# 외국인(마,일,중)이 국내관광지(5곳) 방문 관련자료 사용 
# 나라별 관광지 상관관계 확인하기
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib

# 산점도 그리기 함수
def setScatterGraph(tour_table, all_table, tourpoint):
    # 계산할 관광지명에 해당하는 자료만 뽑아tour에 저장하고 외국인자료와 병합
    tour = tour_table[tour_table['resNm'] == tourpoint]
    # print(tour)
    merge_table = pd.merge(tour, all_table, left_index=True, right_index=True)
    merge_table.info()
    # <class 'pandas.core.frame.DataFrame'>
    # Index: 69 entries, 201101 to 201609
    # Data columns (total 5 columns):
    #  #   Column  Non-Null Count  Dtype 
    # ---  ------  --------------  ----- 
    #  0   resNm   69 non-null     object
    #  1   ForNum  69 non-null     int64 
    #  2   china   69 non-null     int64 
    #  3   japan   69 non-null     int64 
    #  4   usa     69 non-null     int64 
    # dtypes: int64(4), object(1)
    
    # 시각화 - 상관계수
    fig = plt.figure()
    fig.suptitle(tourpoint + ' 상관관계분석')

    plt.subplot(1,3,1)
    plt.xlabel('중국인 방문수')
    plt.ylabel('외국인 입장객 수')
    lamb1 = lambda p:merge_table['china'].corr(merge_table['ForNum'])
    r1 = lamb1(merge_table)
    plt.title('r={:.5f}'.format(r1))
    plt.scatter(merge_table['china'],merge_table['ForNum'],alpha=0.7,c='red')

    plt.subplot(1,3,2)
    plt.xlabel('일본인 방문수')
    plt.ylabel('외국인 입장객 수')
    lamb2 = lambda p:merge_table['japan'].corr(merge_table['ForNum'])
    r2 = lamb2(merge_table)
    plt.title('r={:.5f}'.format(r2))
    plt.scatter(merge_table['japan'],merge_table['ForNum'],alpha=0.7,c='green')

    plt.subplot(1,3,3)
    plt.xlabel('미국인 방문수')
    plt.ylabel('외국인 입장객 수')
    lamb3 = lambda p:merge_table['usa'].corr(merge_table['ForNum'])
    r3 = lamb3(merge_table)
    plt.title('r={:.5f}'.format(r3))
    plt.scatter(merge_table['usa'],merge_table['ForNum'],alpha=0.7,c='blue')

    plt.tight_layout()
    plt.show()
    plt.close()

    return [tourpoint,r1,r2,r3]

def processFunc():
    # 서울시 관광지 정보 파일
    fname = "ex3_data\서울특별시_관광지입장정보_2011_2016.json"
    jsonTP = json.loads(open(fname,'r',encoding='utf-8').read())
    tour_table = pd.DataFrame(jsonTP, columns=('yyyymm','resNm','ForNum'))      # 연월, 관광지명, 입장객수
    tour_table = tour_table.set_index('yyyymm')
    # print(tour_table)
    resNm = tour_table.resNm.unique()
    # print('resNm : ', resNm[:5])        ['창덕궁' '운현궁' '경복궁' '창경궁' '종묘']

    # 중국인 관광객 정보 파일 읽기
    cdf = "ex3_data\중국인방문객.json"
    cdata = json.loads(open(cdf,'r',encoding='utf-8').read())
    china_table = pd.DataFrame(cdata, columns=('yyyymm', 'visit_cnt'))
    china_table = china_table.rename(columns={'visit_cnt':'china'})
    china_table = china_table.set_index('yyyymm')
    # print(china_table[:5])

    # 일본인 관광객 정보 파일 읽기
    jdf = "ex3_data\일본인방문객.json"
    jdata = json.loads(open(jdf,'r',encoding='utf-8').read())
    japan_table = pd.DataFrame(jdata, columns=('yyyymm', 'visit_cnt'))
    japan_table = japan_table.rename(columns={'visit_cnt':'japan'})
    japan_table = japan_table.set_index('yyyymm')
    # print(japan_table[:5])

    # 미국인 관광객 정보 파일 읽기
    udf = "ex3_data\미국인방문객.json"
    udata = json.loads(open(udf,'r',encoding='utf-8').read())
    usa_table = pd.DataFrame(udata, columns=('yyyymm', 'visit_cnt'))
    usa_table = usa_table.rename(columns={'visit_cnt':'usa'})
    usa_table = usa_table.set_index('yyyymm')
    # print(usa_table[:5])

    # 3국 관광객 통합 데이터
    all_table = pd.merge(china_table, japan_table, left_index=True, right_index=True)
    all_table = pd.merge(all_table, usa_table, left_index=True, right_index=True)
    all_table.info()
    # <class 'pandas.core.frame.DataFrame'>
    # Index: 72 entries, 201101 to 201612
    # Data columns (total 3 columns):
    # #   Column  Non-Null Count  Dtype
    # ---  ------  --------------  -----
    # 0   china   72 non-null     int64
    # 1   japan   72 non-null     int64
    # 2   usa     72 non-null     int64
    # dtypes: int64(3)

    r_list = []
    for tourpoint in resNm[:5]:
        r_list.append(setScatterGraph(tour_table, all_table, tourpoint))

    # print(r_list)
    r_df = pd.DataFrame(r_list, columns=['고궁명', '중국', '일본', '미국'])
    r_df = r_df.set_index('고궁명')
    # 상관계수 dataframe 출력
    # print(r_df)

    r_df.plot(kind='bar',rot=50)
    plt.show()


if __name__ == "__main__":
    processFunc()