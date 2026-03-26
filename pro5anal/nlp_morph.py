# nlp_morphing
# 한글 형태소 분석
# 형태소(Morpheme): 의미를 가지는 가장 작은 단위

from konlpy.tag import Okt,Kkma, Komoran

text = "나는 오늘 아침에 지하철을 타고 교육장으로 갔다. 가는 길에 사람이 너무 많아 힘들었다. 밖을 보니 꽃망울이 보였다 봄이 이미 왔나 보다 싶었다."
print("Okt","-"*20)
okt = Okt()
print('형태소 : ', okt.morphs(text))
print('품사 태깅 : ', okt.pos(text))
print('품사 태깅(어간 포함) : ', okt.pos(text, stem=True))
print('명사 추출 : ', okt.nouns(text))

print("Kkma","-"*20)
kkma=Kkma()
print('형태소 : ', kkma.morphs(text))
print('품사 태깅 : ', kkma.pos(text))
print('명사 추출 : ', kkma.nouns(text))

print("Komoran","-"*20)
komoran=Komoran()
print('형태소 : ', komoran.morphs(text))
print('품사 태깅 : ', komoran.pos(text))
print('명사 추출 : ', komoran.nouns(text))
