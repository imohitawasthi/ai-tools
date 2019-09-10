
"""



"""

from functools import partial
from random import random

from aitools.core.utils import gradient_descent
from aitools.utils import mathematics


class LinearRegression:

    def __init__(self):
        self.beta = None

    def build(self, independent_features, dependent_feature):

        base_func = partial(cost, independent_features, dependent_feature)
        gradient_func = partial(log_gradient, independent_features, dependent_feature)

        beta_zero = [random.random() for _ in range(len(independent_features[0]))]
        self.beta = gradient_descent.maximize_batch(base_func, gradient_func, beta_zero)

    def predict(self):
        pass


def price(x, theta): return mathematics.dot(x, theta)


def cost(x, theta, y): return ((price(x, theta) - y)**2).mean()/2


def derive(fn_x, value):
    h = 0.00000000001
    top = fn_x(value + h) - fn_x(value)
    bottom = h
    slope = top / bottom
    # Returns the slope to the third decimal
    return float("%.3f" % slope)


X = [
    [1000, 100, 10],
    [2000, 200, 20],
    [3000, 300, 30],
    [4000, 400, 40],
]

y = [
    1, 2, 3, 4
]
