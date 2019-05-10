"""
    The Logistic Regression is a statistical model in which a logistic function is user to model a binary
    dependent variable(One of the most simplest form of Logistic function).

    Here the logistic function is a sigmoid function.
"""

from functools import partial, reduce
from aitools.utils import mathematics
from aitools.core.utils import gradient_descent
import math
import random


class LogisticRegression:

    def __init__(self):
        self.beta = None

    def build(self, independent_features, dependent_feature):
        """

        :param independent_features:
        :param dependent_feature:
        :return:
        """
        base_func = partial(log_likelihood, independent_features, dependent_feature)
        gradient_func = partial(log_gradient, independent_features, dependent_feature)

        beta_zero = [random.random() for _ in range(len(independent_features[0]))]
        self.beta = gradient_descent.maximize_batch(base_func, gradient_func, beta_zero)

    def predict(self, independent_features):

        likelihood = []

        for features in independent_features:
            likelihood.append({
                'feature': features,
                'likelihood': mathematics.sigmoid(mathematics.dot(self.beta, features))
            })

        return likelihood


def log_likelihood(x, y, beta):
    return sum(
        log_likelihood_i(x_i, y_i, beta)
        for x_i, y_i in zip(x, y)
    )


def log_likelihood_i(x_i, y_i, beta):
    if y_i == 1:
        return math.log(mathematics.sigmoid(mathematics.dot(x_i, beta)))
    else:
        return math.log(1 - mathematics.sigmoid(mathematics.dot(x_i, beta)))


def log_gradient(x, y, beta):
    return reduce(
        mathematics.vector_add,
        [log_gradient_i(x_i, y_i, beta) for x_i, y_i in zip(x, y)]
    )


def log_gradient_i(x_i, y_i, beta):
    """the gradient of the log likelihood
    corresponding to the ith data point"""
    return [log_partial_ij(x_i, y_i, beta, j) for j, _ in enumerate(beta)]


def log_partial_ij(x_i, y_i, beta, j):
    """here i is the index of the data point,
    j the index of the derivative"""
    return (y_i - mathematics.sigmoid(mathematics.dot(x_i, beta))) * x_i[j]
