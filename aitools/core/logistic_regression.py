"""



"""

from aitools.core import gradient_descent
from aitools.mathematics import mathematics


class LogisticRegression:

    def __init__(
        self,
        independent_variable,
        dependent_variable,
        learning_rate=None,
        iteration_count=100000,
        verbose=False
    ):

        self.independent_variable = independent_variable
        self.dependent_variable = dependent_variable
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        self.verbose = verbose

    # def train(self):
    #     gd = gradient_descent.GradientDescent(
    #         self.cost, self.gradient, self.independent_variable, self.dependent_variable
    #     )
    #     pass

    def cost(self, x):
        return mathematics.sigmoid(self.independent_variable * x)

    def gradient(self, x):
        return mathematics.sigmoid(self.independent_variable * x) * (1 - mathematics.sigmoid(self.independent_variable * x))

    def fit(self):
        gd = gradient_descent.GradientDescent(
            self.cost, self.gradient, self.independent_variable, self.dependent_variable
        )
        pass


