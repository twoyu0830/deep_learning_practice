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
df = pd.read_csv("C:/Users/ygyg0/Google 드라이브/python_practice/deep_learning_practice/deeplearning_documents/dataset/iris.csv", names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
print(df.head())

# +
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue = 'species'); #상관도 그래프(두 개씩 짝지어서)
plt.show()

# +
#데이터 안에 문자열이 포함돼 있으므로 pandas 라이브러리로 데이터 불러옴
df = pd.read_csv("C:/Users/ygyg0/Google 드라이브/python_practice/deep_learning_practice/deeplearning_documents/dataset/iris.csv", names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

dataset = df.values
X = dataset[:, 0:4].astype(float)
Y_obj = dataset[:, 4] #문자열을 숫자로 바꾸기 전이므로 임시 저장소임

# +
from sklearn.preprocessing import LabelEncoder #클래스 이름을 숫자로 바꿔주기

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj) #진짜 Y저장소

# +
from tensorflow.keras.utils import np_utils #숫자들을 0, 1로 이루어진 이진수 비슷하게 바꿔줌(활성화 함수에 들어갈 수 있게)

Y_encoded = tf.keras.utils.to_categotical(Y)
# -

model = Sequential()
model.add(Dense(16, imput_dim = 4, activation = 'relu'))
#결과가 3개 중 하나이므로 출력층 노트가 3개, softmax함수: 합해서 1이되게 함, 원-한-인코딩이 쉽게 되게 함
model.add(Dense(3, activation = 'softmax')) 
