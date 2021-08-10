# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
df = pd.read_csv("C:/Users/ygyg0/Google 드라이브/python_practice/deep_learning_practice/deeplearning_documents/dataset/pima-indians-diabetes.csv", names = ["pregnant", "plasma", "pressure", "thickness", "insulin", "BMI", "pedigree", "age", "class"])


print(df.head(5)) #데이터의 몇 줄 불러올 것인지

print(df.info())

print(df.describe()) #정보별 특징(샘플 수(count), 평균(mean), 표준편차(std), 최솟값(min), 백분위(%), 최댓값(max))

print(df[['pregnant', 'class']]) #데이터의 일부 컬럼(열)만 보고 싶을 때

#데이터 한번 더 가공하기(임신 횟수 당 당뇨병 발병 확률)
print(df[['pregnant', 'class']].groupby(['pregnant'], as_index = False).mean().sort_values(by = 'pregnant', ascending = True))

# +
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (12, 12)) #그래프의 크기 설정
#두 항목(x, y축 항목)간의 상관관계 나타내기(상관관계가 높을수록 밝은 색)
sns.heatmap(df.corr(), linewidth = 0.1, vmax = 0.5, cmap = plt.cm.gist_heat, linecolor = 'white', annot = True)
plt.show()
# -

#위에서 상관관계가 가장 높았던 plasma와 class 항목만 떼어서 관계를 그래프로 나타나게함
grid = sns.FacetGrid(df, col = 'class')
grid.map(plt.hist, 'plasma', bins = 10)
plt.show()

# +
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

#seed값 생성
np.random.seed(3)
tf.random.set_seed(3)

#데이터 로드
dataset = numpy.loadtxt("C:/Users/ygyg0/Google 드라이브/python_practice/deep_learning_practice/deeplearning_documents/dataset/pima-indians-diabetes.csv", delimiter = ",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

#모델의 설정
model = Sequential()
model.add(Dense(12, input_dim = 8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#모델 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#모델 실행
model.fit(X, Y, epochs = 200, batch_size = 10)

#결과 출력
print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
