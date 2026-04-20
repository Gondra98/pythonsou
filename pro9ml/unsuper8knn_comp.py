# 지도학습(KNN) / 비지도학습(KMeans) 비교 - iris dataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()
train_x, test_x, train_y, test_y = train_test_split(iris_data['data'], iris_data['target'], test_size=0.25, random_state=42)

print(train_x.shape, test_x.shape, train_y.shape, test_y.shape) 
# (112, 4) (38, 4) (112,) (38,)

print('지도학습 - K최근접 이웃 알고리즘')
from sklearn.neighbors import KNeighborsClassifier
knnModel = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
knnModel.fit(train_x, train_y)

# 예측 및 성능 확인(acc)
pred_label = knnModel.predict(test_x)
print('예측값 : ', pred_label[:10])     # [1 0 2 1 1 0 1 2 1 1]
print('실제값 : ', test_y[:10])         # [1 0 2 1 1 0 1 2 1 1]

from sklearn.metrics import accuracy_score
print('분류 정확도 : ', accuracy_score(test_y, pred_label)) # 1.0

# 새로운 값 분류
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
print(f'KNN : 새로운 값은 {knnModel.predict(new_input)}로 분류됨')
# KNN : 새로운 값은 [1]로 분류됨

# 새로운 데이터는 몇번째 자료와 거리를 확인했을까?
dist, index = knnModel.kneighbors(new_input)
print(dist, index)
# [[0.2236068  0.3        0.43588989]] [[71 82 31]] 
# k=3이므로 71, 82, 31번 자료가 분류에 참여함. 새로운 자료와 거리는 0.223, 0.3, 0.435

print('\n비지도학습 - K평균 군집 알고리즘')
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters=3, init='k-means++', random_state=0)
kmeansModel.fit(train_x)    # label이 주어지지 않는다.
print(kmeansModel.labels_)  # 군집 라벨 출력
print()
# 군집별 자료보기
print('0 cluster : ', train_y[kmeansModel.labels_ == 0])    # 0번째 군집은 라벨2
print('1 cluster : ', train_y[kmeansModel.labels_ == 1])    # 1번째 군집은 라벨0
print('2 cluster : ', train_y[kmeansModel.labels_ == 2])    # 2번째 군집은 라벨1
# 0 cluster :  [2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2]
# 1 cluster :  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# 2 cluster :  [2 1 1 1 2 1 1 1 1 1 2 1 1 1 2 2 2 1 1 1 1 1 2 1 1 1 1 2 1 1 1 2 1 1 1 1 1
#  1 1 1 1 1 1 2 2 1 2 1]

# 새로운 값 군집 분류
new_input = np.array([[6.1, 2.8, 4.7, 1.2]])
clu_pred = kmeansModel.predict(new_input)
print(f'KMeans : 새로운 값은 군집 {clu_pred}로 분류됨')
# KMeans : 새로운 값은 군집 [2]로 분류됨

print()
# 군집 모델 성능 파악
pred_cluster = kmeansModel.predict(test_x)
print('예측값 : ', pred_cluster)    # [2 1 0 2 2 1 2 0 2 2 0 1 1 1 1 2 0 2 2 0 1 2 1 0 0 0 0 0 1 1 1 1 2 1 1 2 2 1]

# 평가 데이터를 적용해 예측한 군집을, 각 iris의 종류를 의미하는 라벨값으로
# 다시 바꿔줘야 실제 라벨과 비교해 성능 측정 가능
np_arr = np.array(pred_cluster)
print('np_arr : ', np_arr)
np_arr[np_arr == 0], np_arr[np_arr == 1], np_arr[np_arr == 2] = 3, 4, 5     # 임시 저장용
np_arr[np_arr == 3] = 2     # 군집 3을 2(versicolor)로 변경
np_arr[np_arr == 4] = 0     # 군집 4을 0(setosa)로 변경
np_arr[np_arr == 5] = 1     # 군집 5을 1(verginicar)로 변경
print('np_arr : ', np_arr)

predict_label = np_arr.tolist()
print(predict_label)
print(f'군집 test acc : {np.mean(predict_label == test_y)}')    # 0.9473684210526315






