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
