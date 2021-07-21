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

#가중치, 바이어스
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1

#퍼셉트론(가중합&활성화함수)
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1

#n1
def NAND(x1, x2):
    return MLP(np.array([x1, x2]), w11, b1)

#n2
def OR(x1, x2):
    return MLP(np.array([x1, x2]), w12, b2)

#y(out)
def AND(x1, x2):
    return MLP(np.array([x1, x2]), w2, b2)

#최종값
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))

if __name__ == '__main__':
    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y = XOR(x[0], x[1])
        print("입력 값: " + str(x) + "출력 값: " + str(y))
