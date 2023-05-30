"""Class 10, Exercise A: Equation Discovery with Linear Regression"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

import data_generators
from aml import fancy_print


DATA_GENERATOR = data_generators.generate_conservation_of_energy


# 1 Implement linear regression
def linear_regression(X, y, epsilon=1e-2, ridge_param=None, lasso_param=None):

    variables = X.columns

    if ridge_param and lasso_param:
        raise ValueError("Kwargs 'ridge_parameter' and 'lasso_parameter' cannot both differ from None!")

    elif ridge_param or not any([ridge_param, lasso_param]):
        regularisation_factor = (ridge_param or 0) * np.eye(X.shape[1])
        beta = np.linalg.pinv(X.T.dot(X) + regularisation_factor).dot(X.T).dot(y)

    else:
        beta = minimize(
            lambda b: np.sum((X.dot(b)-y)**2) + lasso_param*np.sum(np.abs(b)), np.random.random(X.shape[1]))["x"]

    beta = np.where(beta > epsilon, beta, 0)
    equation = " ".join([f"{b:+.2f}*{variable}" for b, variable in zip(beta, variables) if b != 0])

    return equation


def preprocess_for_equation_discovery(data, target_name, degree=2):

    X, y = data.drop(columns=target_name), data[target_name]

    poly = PolynomialFeatures(degree=degree)
    X = poly.fit_transform(X, y)
    feature_names = [name.replace(" ", "") for name in poly.get_feature_names_out()]
    X = pd.DataFrame(X, columns=feature_names)

    return X, y


# 2 Test linear regression on given data
fancy_print("LINEAR REGRESSION")
noises = [0, 0.001, 0.01, 0.1]
for noise in noises:
    features, target = preprocess_for_equation_discovery(
        data=DATA_GENERATOR(100, noise=noise),
        target_name="E",
        degree=3)
    fancy_print(f"Noise {noise}", linear_regression(features, target))
del noise, noises


# 3 Handle noise with ridge regression
fancy_print("RIDGE REGRESSION")
lambdas = [None, 0.1, 1, 10]
for lambda_ in lambdas:
    features, target = preprocess_for_equation_discovery(
        data=DATA_GENERATOR(100, noise=0.01),
        target_name="E",
        degree=3)
    fancy_print(f"Lambda {lambda_}", linear_regression(features, target, ridge_param=lambda_))
del lambda_, lambdas


# 4 Handle noise with lasso regression
fancy_print("LASSO REGRESSION")
lambdas = [None, 0.1, 1, 10]
for lambda_ in lambdas:
    features, target = preprocess_for_equation_discovery(
        data=DATA_GENERATOR(100, noise=0.01),
        target_name="E",
        degree=3)
    fancy_print(f"Lambda {lambda_}", linear_regression(features, target, lasso_param=lambda_))
del lambda_, lambdas
