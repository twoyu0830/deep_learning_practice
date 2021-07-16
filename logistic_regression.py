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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#공부한 시간과 합격 여부
data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
x_data = [[i[0] for i in data]]
y_data = [[i[1] for i in data]]

#그래프 표현
plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)

#시그모이드 함수 안의 a, b값 초기화
a = 0
b = 0

#학습률
lr = 0.05

#학습 몇 번 반복할지
epochs = 2001

#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.e ** -(x))

#경사하강법
for i in range(epochs):
    for x_data, y_data in data:
        a_diff = x_data * (sigmoid(a * x_data + b) - y_data)
        b_diff = sigmoid(a * x_data + b) - y_data
        a -= lr * a_diff
        b -= lr * b_diff
        if i % 1000 == 0:
            print("epoch = %.f, a = %.04f, b = %.04f, a_diff = %.04f, b_diff = %.04f" % (i, a, b, a_diff, b_diff))

#그래프 표현
plt.scatter(x_data, y_data)
plt.xlim(0, 15)
plt.ylim(-.1, 1.1)
x_range = (np.arange(0, 15, 0.1))
plt.plot(np.arange(0, 15, 0.1), np.array([sigmoid(a * x + b) for x in x_range]))
plt.show()
