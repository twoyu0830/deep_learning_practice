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

#공부한 시간, 실제 점수 데이터
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]

#그래프 표현
plt.figure(figsize = (8, 5))
plt.scatter(x, y)
plt.show()

#리스트를 뒤에서 연산에 사용하기 위함
x_data = np.array(x)
y_data = np.array(y)


#기울기, y절편 초기화
a = 0
b = 0

#학습률
lr = 0.03

#학습 몇 번 반복할지
epochs = 2001

#경사하강법
for i in range(epochs):
    y_pred = a * x_data + b
    error = y_data - y_pred
    a_diff = -(2 / len(x_data)) * sum(error * x_data)
    b_diff = -(2 / len(x_data)) * sum(error)
    
    a -= lr * a_diff #a값 좌표 조절(이차곡선의 x축에서)
    b -= lr * b_diff
    
    if i % 100 == 0: #100번 반복될 때마다 상태 출력
        print("epochs = %.f, a = %.04f, b = %.04f, a_diff = %.04f, b_diff = %.04f" % (i, a, b, a_diff, b_diff))
        
#최종 그래프(선형회귀)
y_pred = a * x_data + b
plt.scatter(x, y)
plt.plot([min(x_data), max(x_data)], [min(y_pred), max(y_pred)])
plt.show()

#최종 방정식
print("y = %.01fx + %.00f" % (a, b))
