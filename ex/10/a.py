"""Class 10, Exercise A: Equation Discovery with Linear Regression"""

import numpy as np


# 1 Define a linear regression function
def linear_regrassion(x, y, epsilon=1e-2):

    variables = x.columns

    beta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    beta = np.where(beta > epsilon, beta, 0)

    equation = " + ".join([f"({b}{variable})" for b, variable in zip(beta, variables) if b != 0])

    return equation
