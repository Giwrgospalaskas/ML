import math
import numpy as np
import pandas as pd
from sklearn import linear_model


df = pd.read_csv('test_scores.csv')

y = np.array(df.cs)
x = np.array(df.math)

def sklearn_way():
    df = pd.read_csv('test_scores.csv')
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']], df.cs)
    return reg.coef_, reg.intercept_


def gradient_descent(x, y):
    m = 0
    b = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    cost_previous = 0

    for i in range(iterations):
        y_predicted = m*x + b
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        m = m-learning_rate*md
        b = b- learning_rate*bd
        if math.isclose(cost,cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print(f"m {m}, b {b}, cost {cost}, iteration {i}")

gradient_descent(x,y)
coef , inter = sklearn_way()
print(coef , inter)
