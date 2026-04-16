'''
[GaussanNB 문제] 
독버섯(poisonous)인지 식용버섯(edible)인지 분류
https://www.kaggle.com/datasets/uciml/mushroom-classification
feature는 중요변수를 찾아 선택, label:class
참고 : from xgboost import plot_importance


데이터 변수 설명 : 총 23개 변수가 사용됨.

여기서 종속변수(반응변수)는 class 이고 나머지 22개는 모두 입력변수(설명변수, 예측변수, 독립변수).
변수명 변수 설명
class      edible = e, poisonous = p
cap-shape    bell = b, conical = c, convex = x, flat = f, knobbed = k, sunken = s
cap-surface  fibrous = f, grooves = g, scaly = y, smooth = s
cap-color     brown = n, buff = b, cinnamon = c, gray = g, green = r, pink = p, purple = u, red = e, white = w, yellow = y
bruises        bruises = t, no = f
odor            almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
gill-attachment attached = a, descending = d, free = f, notched = n
gill-spacing close = c, crowded = w, distant = d
gill-size       broad = b, narrow = n
gill-color      black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e, white = w, yellow = y
stalk-shape  enlarging = e, tapering = t
stalk-root    bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing = ?
stalk-surface-above-ring fibrous = f, scaly = y, silky = k, smooth = s
stalk-surface-below-ring fibrous = f, scaly = y, silky = k, smooth = s
stalk-color-above-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
stalk-color-below-ring brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
veil-type      partial = p, universal = u
veil-color     brown = n, orange = o, white = w, yellow = y
ring-number none = n, one = o, two = t
ring-type     cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
spore-print-color black = k, brown = n, buff = b, chocolate = h, green = r, orange =o, purple = u, white = w, yellow = y
population abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
habitat       grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_importance

df = pd.read_csv('mushrooms.csv')
print(df.head(3))
#   class cap-shape cap-surface cap-color bruises odor gill-attachment gill-spacing  ... stalk-color-below-ring veil-type veil-color ring-number ring-type spore-print-color population habitat
# 0     p         x           s         n       t    p               f            c  ...                      w         p          w           o         p                 k          s       u      
# 1     e         x           s         y       t    a               f            c  ...                      w         p          w           o         p                 n          n       g      
# 2     e         b           s         w       t    l               f            c  ...                      w         p          w           o         p                 n          n       m 
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 8124 entries, 0 to 8123
# Data columns (total 23 columns):
#  #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   class                     8124 non-null   object
#  1   cap-shape                 8124 non-null   object
#  2   cap-surface               8124 non-null   object
#  3   cap-color                 8124 non-null   object
#  4   bruises                   8124 non-null   object
#  5   odor                      8124 non-null   object
#  6   gill-attachment           8124 non-null   object
#  7   gill-spacing              8124 non-null   object
#  8   gill-size                 8124 non-null   object
#  9   gill-color                8124 non-null   object
#  10  stalk-shape               8124 non-null   object
#  11  stalk-root                8124 non-null   object
#  12  stalk-surface-above-ring  8124 non-null   object
#  13  stalk-surface-below-ring  8124 non-null   object
#  14  stalk-color-above-ring    8124 non-null   object
#  15  stalk-color-below-ring    8124 non-null   object
#  16  veil-type                 8124 non-null   object
#  17  veil-color                8124 non-null   object
#  18  ring-number               8124 non-null   object
#  19  ring-type                 8124 non-null   object
#  20  spore-print-color         8124 non-null   object
#  21  population                8124 non-null   object
#  22  habitat                   8124 non-null   object
# dtypes: object(23)
# memory usage: 1.4+ MB
# None


feature = df.drop('class', axis=1)
print(feature.head(3), feature.shape)   # (8124, 22)

for col in feature.columns:
    feature[col] = feature[col].astype('category').cat.codes
print(feature.head(3), feature.shape)   # (8124, 22)


label = df['class']
print(label.head(3), label.shape)   # (8124,)
label = df['class'].map({'e':1, 'p':0})
print(label.head(3), label.shape)


x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=42, stratify=label)

print(x_train.shape, x_test.shape)  # (5686, 22) (2438, 22)

xgb_clf = xgb.XGBClassifier(
    booster='gbtree',      
    max_depth=6,           
    n_estimators=200,      
    eval_metric='logloss', 
    random_state=42
)

xgb_clf.fit(x_train, y_train)
xgb_pred = xgb_clf.predict(x_test)
print("분류 정확도 : ", accuracy_score(y_test, xgb_pred))

print('예측값 : ', xgb_pred[:5])
print('실제값 : ', y_test[:5].values)


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_importance(xgb_clf, ax=ax)
plt.show()


# 1. 중요도 상위 feature 추출
importance_dict = xgb_clf.get_booster().get_fscore()
importance_df = pd.DataFrame({
    'feature': list(importance_dict.keys()),
    'importance': list(importance_dict.values())
}).sort_values('importance', ascending=False)

# 2. 상위 N개 선택 (plot_importance 보고 결정)
top_features = importance_df['feature'][:10].tolist()

# 3. GaussianNB 학습
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

nb_model = GaussianNB()
nb_model.fit(x_train[top_features], y_train)

# 4. 평가
pred = nb_model.predict(x_test[top_features])
print('정확도 : ', accuracy_score(y_test, pred))    # 0.9122231337161608
print(classification_report(y_test, pred, target_names=['poisonous', 'edible']))
#               precision    recall  f1-score   support

#    poisonous       0.94      0.88      0.91      1175
#       edible       0.89      0.94      0.92      1263

#     accuracy                           0.91      2438
#    macro avg       0.91      0.91      0.91      2438
# weighted avg       0.91      0.91      0.91      2438