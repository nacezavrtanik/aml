"""Class 10, Exercise A: Equation Discovery with Linear Regression"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from vajeED_1_podatki import generiraj_energijski_zakon
from aml import fancy_print


# 1 Define a linear regression function
def linear_regrassion(X, y, epsilon=1e-2, ridge_parameter=0):

    variables = X.columns

    regularisation_factor = ridge_parameter * np.eye(X.shape[1])

    beta = np.linalg.pinv(X.T.dot(X) + regularisation_factor).dot(X.T).dot(y)
    beta = np.where(beta > epsilon, beta, 0)

    equation = " ".join([f"{b:+.2f}*{variable}" for b, variable in zip(beta, variables) if b != 0])

    return equation


def preprocess_for_equation_discovery(data, target_name, degree=2):

    X, y = data.drop(columns=target_name), data[target_name]

    poly = PolynomialFeatures(degree=degree)
    X = poly.fit_transform(X, y)
    feature_names = poly.get_feature_names_out()
    X = pd.DataFrame(X, columns=feature_names)

    return X, y


# 2 Test the function on data for the energy conservation law
fancy_print("LINEAR REGRESSION")
noises = [0, 0.001, 0.01, 0.1]
for noise in noises:
    features, target = preprocess_for_equation_discovery(
        data=generiraj_energijski_zakon(100, noise=noise),
        target_name="E",
        degree=3)
    fancy_print(f"Noise {noise}", linear_regrassion(features, target))
del noise, noises


# 3 Handle noise with regularisation
fancy_print("RIDGE REGRESSION")
lambdas = [0, 0.1, 1, 10]
for lambda_ in lambdas:
    features, target = preprocess_for_equation_discovery(
        data=generiraj_energijski_zakon(100, noise=0.01),
        target_name="E",
        degree=3)
    fancy_print(f"Lambda {lambda_}", linear_regrassion(features, target, ridge_parameter=lambda_))
del lambda_, lambdas
