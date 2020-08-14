import numpy as np

def sigmoid(x):
    return 1 / (1 + 1 / np.e ** x)

def calc_delta(o, t):
    return (o - t) * o * (1 - o)

def calc_err(t, y):
    return ((t - y) ** 2) / 2
