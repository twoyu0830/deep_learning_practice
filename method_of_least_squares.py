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

#x값과 y값
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]
print("x = {0}, y = {1}".format(x, y))

#x값과 y값의 평균
mx = np.mean(x)
my = np.mean(y)
print("mx = {0}, my = {1}".format(mx, my))

#기울기 a의 분모
def bottom(x, mx):
    t = 0
    for i in x:
        t += (i - mx) ** 2
    return t
divisor = bottom(x, mx)

#기울기 a의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)

print("a의 분모: ", divisor)
print("a의 분자: ", dividend)

#기울기 a, y절편 b
a = dividend / divisor
b = my - (mx * a)
print("a = {0}, b = {1}".format(a, b))

#선형 회귀 방정식 완성
print("y = {0}x + {1}".format(a, b))

#선형 회귀 방정식 이용해 실제 성적 예측하는 함수
def l_r(t):
    s = a * t + b
    print("이 공부시간으로는 {0}점을 맞습니다.".format(s))
    
#실제 성적
l_r(9) 
