# 스팸 메일 분류기 - spam 자료를 파일에서 읽기
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import koreanize_matplotlib

df= pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/mydata.csv')
print(df.head(3), df.shape)
#               text label
# 0    광고성 메일을 확인하세요  spam
# 1      회의 일정 변경 공지   ham
# 2  무료 쿠폰을 지금 사용하세요  spam (20, 2)
print(df.info())
# RangeIndex: 20 entries, 0 to 19
# Data columns (total 2 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   text    20 non-null     object
#  1   label   20 non-null     object
# dtypes: object(2)

df['label'] = df['label'].str.strip().str.lower()
texts = df['text'].tolist()
labels = df['label'].tolist()
print(texts[:3])
# ['광고성 메일을 확인하세요', '회의 일정 변경 공지', '무료 쿠폰을 지금 사용하세요']
print(labels[:3])
# ['spam', 'ham', 'spam']

x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42, stratify=labels)

# 벡터화
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train) # 단어 사전을 만듦
x_test_vec = vectorizer.transform(x_test)       # test는 transform만 함 (숫자 변환만)

model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

acc = accuracy_score(y_test, y_pred)
print('분류 정확도 : ', acc)    # 0.8

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
print(cm)

# Confusion matrix를 시각화
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot(cmap='Blues')
plt.title('Confusion matrix(혼동행렬)')
plt.show()

# 사용자 입력 메일 내용 분류
while True:
    userInput = input('이메일 내용 입력(종료는 q):')
    if userInput.lower() == 'q':
        break
    x_new = vectorizer.transform([userInput])
    prob = model.predict_proba(x_new)[0]
    spam_prob = prob[model.classes_.tolist().index('spam')]

    result = '스팸이에요' if spam_prob >= 0.7 else "정상 메일입니다"
    print(f'스팸확률은 {spam_prob:.2f} -> {result}')

# 이메일 내용 입력(종료는 q):광고성 메일을 확인하세요
# 스팸확률은 0.93 -> 스팸이에요
# 이메일 내용 입력(종료는 q):회의 일정 변경 공지
# 스팸확률은 0.05 -> 정상 메일입니다
# 이메일 내용 입력(종료는 q):무료 쿠폰 보냈습니다 지금 확인하세요
# 스팸확률은 0.99 -> 스팸이에요
