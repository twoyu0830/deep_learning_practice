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
#여기 있는 코드들은 경사하강법을 이용한 선형 회귀에서 축을 단지 하나 추가한 다중 선형 회귀를 표현한 것임.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#공부한 시간, 과외 횟수, 실제 점수 데이터
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [i[0] for i in data]
x2 = [i[1] for i in data]
y = [i[2] for i in data]

#그래프 표현
ax = plt.axes(projection = '3d')
ax.set_xlabel('study_hours')
ax.set_ylabel('private_class')
ax.set_zlabel('Score')
ax.dist = 11
ax.scatter(x1, x2, y)
plt.show()

#리스트를 뒤에서 연산에 사용하기 위함
x1_data = np.array(x1)
x2_data = np.array(x2)
y_data = np.array(y)

#기울기 2개, y절편 초기화
a1 = 0
a2 = 0
b = 0

#학습률
lr = 0.02

#학습 볓 번 반복할지
epochs = 2001

#경사하강법
for i in range(epochs):
    y_pred = a1 * x1_data + a2 * x2_data + b
    error = y_data - y_pred
    a1_diff = -(2 / len(x1_data)) * sum(error * x1_data)
    a2_diff = -(2 / len(x2_data)) * sum(error * x2_data)
    b_diff = -(2 / len(y_data)) * sum(error)
    a1 -= lr * a1_diff
    a2 -= lr * a2_diff
    b -= lr * b_diff
    
    if i % 100 == 0: #100번 반복될 때마다 상태 출력
        print("epochs = %.f, a1 = %.04f, a2 = %.04f, b = %.04f, a1_diff = %.04f, a2_diff = %.04f, b_diff = %.04f" % (i, a1, a2, b, a1_diff, a2_diff, b_diff))

#최종 방정식
print("y = %.01fx1 + %.01fx2 + %.00f" % (a1, a2, b))
