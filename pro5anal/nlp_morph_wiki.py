# 위키백과 세종 웹 문서를 읽어 형태소 분석 연습
import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
from urllib import parse        # 한글 인코딩 해주는 라이브러리

okt = Okt()

para = parse.quote("세종")      # 한글 인코딩
url = "https://ko.wikipedia.org/wiki/" + para
# print(url)

headers = {"User-Agent":"Mozilla/5.0"}      # chrome 계열의 브라우저
response = requests.get(url, headers=headers)

if response.status_code == 200:
    page = response.text
    # print(page, type(page))           # str type text 넘어옴
    soup = BeautifulSoup(page, 'lxml')  # bs로 바꿔줌
    
    word_list = []                      # 한글 명사만 저장하는 list
    for item in soup.select("#mw-content-text p"):
        if item.string != None:
            word_list += okt.nouns(item.string)
    print(word_list)
    print('단어수 : ', len(word_list))
    print('중복 제거 후 단어수 : ', len(set(word_list)))
    print()
    word_dict = {}      # 단어의 발생 횟수를 dict로 저장
    for i in word_list:
        if i in word_dict:
            word_dict[i] += 1
        else:
            word_dict[i] = 1
    print('단어당 발생 수', word_dict)
    seri_list = pd.Series(word_list)
    print(seri_list[:3])
    print(seri_list.value_counts()[:5])
    seri_dict = pd.Series(word_dict)
    print(seri_dict[:3])
    print(seri_dict.value_counts()[:5])

    print("dataframe으로 출력")
    df1 = pd.DataFrame(word_list, columns=['단어'])
    print(df1.head(3))
    df2 = pd.DataFrame([word_dict.keys(), word_dict.values()])
    df2 = df2.T
    df2.columns = ['단어','빈도수']
    print(df2.head(3))
    df2.to_csv('nlp_morph_wiki.csv', index=False)

df3 = pd.read_csv('nlp_morph_wiki.csv')
print(df3.head(3))
