# [로지스틱 분류분석 문제3]
# Kaggle.com의 https://www.kaggle.com/truesight/advertisingcsv  file을 사용
# 얘를 사용해도 됨   'testdata/advertisement.csv' 

# 참여 칼럼 : 
#    - Daily Time Spent on Site : 사이트 이용 시간 (분)
#    - Age : 나이,
#    - Area Income : 지역 소득,
#    - Daily Internet Usage :일별 인터넷 사용량(분),
#    - Clicked Ad : 광고 클릭 여부 ( 0 : 클릭x , 1 : 클릭o )
# 광고를 클릭('Clicked on Ad')할 가능성이 높은 사용자 분류.
# 데이터 간 단위가 큰 경우 표준화 작업을 시도한다.
# 모델 성능 출력 : 정확도, 정밀도, 재현율, ROC 커브와 AUC 출력
# 새로운 데이터로 분류 작업을 진행해 본다.


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib

# =============================================
# 1. 데이터 로드 및 전처리
# =============================================
data = pd.read_csv("advertising.csv")

# 분석에 불필요한 문자형/시간형 컬럼 제거
adver_data = data.drop(['Ad Topic Line', 'City', 'Male', 'Country', 'Timestamp'], axis=1)

x = adver_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y = adver_data['Clicked on Ad'].values

# =============================================
# 2. 표준화 (StandardScaler)
# =============================================
# 컬럼별 값의 범위가 다를 때 (예: Age=19~61, Area Income=13000~79000)
# 큰 숫자에 모델이 편향되지 않도록 평균=0, 표준편차=1로 스케일 통일
# ※ y(종속변수)는 0/1 범주형이므로 표준화 대상 아님

sc = StandardScaler()
x = sc.fit_transform(x)
# fit   : 평균, 표준편차 계산 (훈련)
# transform : 실제 변환 적용
# fit_transform : 두 작업을 한번에

# =============================================
# 3. 모델 학습
# =============================================
model = LogisticRegression().fit(x, y)
y_hat = model.predict(x)

# =============================================
# 4. 혼동행렬 & 성능 지표
# =============================================
print(confusion_matrix(y, y_hat))
# [[TP  FN]     TP=489 : 클릭X → 클릭X 예측 (정답)
#  [FP  TN]]    FN=11  : 클릭X → 클릭O 예측 (오탐)
#               FP=21  : 클릭O → 클릭X 예측 (미탐)
#               TN=479 : 클릭O → 클릭O 예측 (정답)

# TP=489, FN=11, FP=21, TN=479
acc         = (489 + 479) / 1000   # (TP+TN) / 전체        → 전체 정확도
recall      = 489 / (489 + 11)     # TP / (TP+FN)          → 실제 양성 중 맞힌 비율 (TPR)
precision   = 489 / (489 + 21)     # TP / (TP+FP)          → 양성 예측 중 실제 양성 비율
specificity = 479 / (479 + 21)     # TN / (TN+FP)          → 실제 음성 중 맞힌 비율
fallout     = 21  / (21  + 479)    # FP / (FP+TN) = 1-특이도 → 위양성률 (FPR)

print(f'정확도   (acc)         : {acc:.3f}')   # 0.968
print(f'재현율   (recall/TPR)  : {recall:.3f}')   # 0.978  → 1에 가까울수록 좋음
print(f'정밀도   (precision)   : {precision:.3f}') # 0.959
print(f'특이도   (specificity) : {specificity:.3f}')# 0.958
print(f'위양성률 (fallout/FPR) : {fallout:.3f}')   # 0.042  → 0에 가까울수록 좋음

# =============================================
# 5. ROC Curve & AUC
# =============================================
# ROC Curve : 모든 threshold(임계값)에서 FPR vs TPR을 그린 곡선
# - threshold 낮추면 → 다 양성으로 예측 → TPR↑ FPR↑ (우상단)
# - threshold 높이면 → 다 음성으로 예측 → TPR↓ FPR↓ (좌하단)
# - 좌상단에 붙을수록 좋은 모델

fpr, tpr, _ = metrics.roc_curve(y, model.decision_function(x))
# decision_function : 각 샘플의 결정함수 값 (양수→클래스1, 음수→클래스0)
# roc_curve : 모든 threshold별 fpr, tpr 배열 반환

auc = metrics.auc(fpr, tpr)
print(f'AUC : {auc:.5f}')  # 0.99196 → 매우 좋은 모델 (0.9↑ : 매우 좋음)

plt.plot(fpr, tpr, 'o-', label=f'LogisticRegression (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='random classifier (AUC=0.5)')
plt.plot([fallout], [recall], 'ro', ms=8, label='현재 모델 위치')  # 현재 FPR, TPR 위치
plt.xlabel('FPR (위양성률)')
plt.ylabel('TPR (재현율)')
plt.title('ROC Curve')
plt.legend()
plt.show()

# =============================================
# 6. 새로운 데이터로 예측
# =============================================
# ※ 학습 때 표준화했으므로 새 데이터도 반드시 같은 sc로 transform 해야 함
print('=' * 40)
DailyTimeSpentonSite = int(input('사이트 이용 시간 (분) : '))
Age                  = int(input('나이 : '))
AreaIncome           = int(input('지역소득 : '))
DailyInternetUsage   = int(input('일별 인터넷 사용량(분) : '))

new_data = pd.DataFrame(
    [[DailyTimeSpentonSite, Age, AreaIncome, DailyInternetUsage]],
    columns=['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']
)
new_data = sc.transform(new_data)  # fit은 하지 않음! 학습 기준 그대로 변환만

print('예측 클래스 : ', model.predict(new_data))         # 0 or 1
print('예측 확률   : ', model.predict_proba(new_data))   # [클릭X 확률, 클릭O 확률]