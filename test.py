# RandomForest는 분류, 회귀 모두 가능. sklearn 모듈은 대개 그러하다.
# 캘리포니아 주택 가격 데이터로 회귀분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('bike_dataset.csv')

print(df.head(3))



