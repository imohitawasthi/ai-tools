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

"""

"""
    
    Notes V1:

        I will try to move forward with more of a generic and a "final product" approach. What ever will come out will be
        the final output of this version
"""


class GradientDescent:

    def __init__(self):
        pass



