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
#딥러닝 구동에 필요한 케라스 함수들 호출
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#필요한 라이브러리들
import numpy as np
import tensorflow as tf

#실행할 때마다 같은 결과 출력하기 위함
np.random.seed(3)
tf.random.set_seed(3)

#수술 환자 데이터
Data_set = np.loadtxt("C:/Users/ygyg0/Google 드라이브/python_practice/deep_learning_practice/deeplearning_documents/dataset/ThoraricSurgery.csv", delimiter = ",")

#환자의 기록, 수술 결과를 X, Y로 구분해 저장
X = Data_set[:, 0:17]
Y = Data_set[:, 17]

#딥러닝 구조 결정
model = sequential()
model.add(Dense(30, input_dim = 17, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#딥러닝 실행
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, Y, epochs = 100, batch_size = 10)

