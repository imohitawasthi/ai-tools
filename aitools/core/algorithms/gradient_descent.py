"""

    Gradient Descent is an optimization technique manly use to find the best parameters for making predictions in our
    modal. The most significant use case of GD is with Linear Regression, Logistic Regression and Half Spaces,
    Support vector machines and, Neural networks

    Basic explanation of GD defines it as "An algorithm to find the minimum value"
        You can imagine this with a person on a hill and trying to find flat ground

    To build GD it takes
        A cost functions whose output has to be minimum
        A derivative of the cost function which is used to calculate the directions and parameters for "Descent"
        Total Iterations algorithm needs to have
        Size of steps taken by GD, usually size is less the 1 to avoid conversion

    More over there are two ways to perform GD
        To iterate over entire data set again and again, this is preferred with smaller data sets - Batch
        To generate small random data sets and iterate over them, this is preferred with larger data sets - Stochastic
        To generate single small random data set and iterate over it - Mini Batch

    -- -- --

    I will try to move forward with more of a generic and a "final product" approach. What ever will come out will be
    the final output of this version

    So there will be a cost function, a gradient function (just a fancy word for derivative), step size and, tolerance
    as default and mandatory parameters to run the algorithm

    User will have the choice over
        Iteration Type
        Variants
        Step size
        Tolerance

    ** all of the above choices will also have a default value.

    Different Iteration Types
        STOCHASTIC - default
        BATCH
        MIN-BATCH

    Different Variants
        VANILLA - default
        MOMENTUM
        ADAGRAD
        ADAM
"""

# imports

import random

from functools import partial

# Constants

TOLERANCE = 0.000001
STEP_SIZES = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
ITERATION = 'STOCHASTIC'
VARIANT = 'VANILLA'


class GradientDescent:

    def __init__(
        self,
        cost,
        gradient,
        independent,
        dependent,
        step_sizes=None,
        tolerance=TOLERANCE,
        iteration=ITERATION,
        variant=VARIANT
    ):

        """
           The Constructor
        :param cost: the cost function
        :param gradient: the derivative of the cost function
        :param independent: independent variables (X)
        :param dependent: dependent variable (y)
        :param step_sizes: step sizes for descent
        :param tolerance: minvalue needs to be achieved
        :param iteration: type of iteration
        :param variant: type of variant
        """

        # As the parameter is immutable
        if step_sizes is None:
            step_sizes = STEP_SIZES

        # Data Set (X, y)
            # Based on "iteration" this will change.
        self.dependent = dependent
        self.independent = independent

        # Cost and gradient functions

        """
            For implementing different iterations here things needs to be changed. 
            This has to be totally calculated on the while loop as in case of different
            algorithms the data-set changes or remains the same.  
        """
        self.cost = partial(cost, self.independent, self.dependent)
        self.gradient = partial(gradient, self.independent, self.dependent)

        # Metadata
        self.step_sizes = step_sizes
        self.tolerance = tolerance
        self.iteration = iteration
        self.variant = variant

    def run(self):
        """

        :return:
        """

        # Safe version of cost function
        cost = safe(self.cost)

        # minimizing
        theta = get_theta_zero(self.independent)
        value = cost(theta)

        """
            This has to be a different function as to implement "different ways of approaching
            the decent".              
        """
        while True:
            # Based on variant this will change
            gradient = self.gradient(theta)
            next_thetas = [
                step(theta, gradient, -step_size) for step_size in self.step_sizes
            ]

            # choose the one that minimizes the error function
            next_theta = min(next_thetas, key=cost)
            next_value = cost(list(next_theta))

            # stop if we're "converging"
            if abs(value - next_value) < self.tolerance:
                return theta
            else:
                theta, value = next_theta, next_value


def get_theta_zero(independent):
    """
        This will get the initial values of theta(Coefficients)
    :return: randomly generated vector
    """
    return [random.random() for _ in range(len(independent[0]))]


def safe(function):
    """
        A function to except a function and does the same work until
        error occurs
    :return:  a new function that's the same as f, except that
            it outputs infinity whenever f produces an error
    """

    def safe_f(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ValueError:
            # float('inf') means "infinity"
            return float('inf')

    return safe_f


def step(theta, direction, step_size):
    """
        move step_size in the direction from theta
    :return: next thetas
    """
    return [theta_i + step_size * direction_i for theta_i, direction_i in zip(theta, direction)]
