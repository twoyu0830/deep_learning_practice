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

#기울기 a, y절편 b(가설)
fake_a_b = [3, 76]

#방정식(가설)
def predict(x):
    return (fake_a_b[0]) * x + (fake_a_b[1])

#공부한 시간, 실제 점수 데이터
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]
    
#predict 함수에 넣은 값들이 될 빈 리스트
predict_result = []

#predict_result의 값들 채우기
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간 = %.f, 실제 점수 = %.f, 예측 점수 = %.f" % (x[i], y[i], predict(x[i])))

#평균 제곱 오차에 들어갈 것 받는 함수
def mse_in(y, predict_result):
    return mse(np.array(y), np.array(predict_result)) #리스트를 연산에 사용하려면 np에 한 번 적용 해줘야함

#평균 제곱 오차
def mse(y, y_hat):
    return ((y - y_hat) ** 2).mean()

print("mse 최종값:", str(mse_in(y, predict_result))) 
