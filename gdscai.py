import numpy as np
import matplotlib.pylot as plt
import pandas as pd
import seaborn as sns


def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        yhat = w * x[i] + b
        cost = cost + (yhat - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        yhat = w * x[i] + b
        dj_dw_temp = (yhat - y[i]) * x[i]
        dj_db_temp = yhat - y[i]
        dj_db += dj_db_temp
        dj_dw += dj_dw_temp
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db
