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

# +
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import tensorflow as tf

#seed값 설정
np.random.seed(3)
tf.random.set_seed(3)

#데이터 입력
df = pd.read_csv('C:/Users/ygyg0/Google 드라이브/deeplearning_documents/dataset/sonar.csv')

#데이터 분류
dataset = df.values
X = dataset[:, 0:60]
Y_obj = dataset[:, 60]

#문자열을 숫자로 변환
e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

#모델 설정
model = Sequential()
model.add(Dense(24, input_dim = 60, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#모델 컴파일
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#모델 실행
model.fit(X, Y, epochs = 200, batch_size = 5)

#결과 출력
print("/n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))
