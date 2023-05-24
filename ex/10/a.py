"""Class 10, Exercise A: Equation Discovery with Linear Regression"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from vajeED_1_podatki import generiraj_energijski_zakon
from aml import fancy_print


# 1 Define a linear regression function
def linear_regrassion(x, y, epsilon=1e-2):

    variables = x.columns

    beta = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    beta = np.where(beta > epsilon, beta, 0)

    equation = " ".join([f"{b:+.2f}*{variable}" for b, variable in zip(beta, variables) if b != 0])

    return equation


# 2 Test the function on data for the energy conservation law
def preprocess_for_equation_discovery(data, target_name, degree=2):

    x, y = data.drop(columns=target_name), data[target_name]

    poly = PolynomialFeatures(degree=degree)
    x = poly.fit_transform(x, y)
    feature_names = poly.get_feature_names_out()
    x = pd.DataFrame(x, columns=feature_names)

    return x, y


noises = [0, 0.001, 0.01, 0.1]

for noise in noises:

    features, target = preprocess_for_equation_discovery(
        data=generiraj_energijski_zakon(100, noise=noise),
        target_name="E",
        degree=3
    )

    fancy_print(f"Equation for noise {noise}", linear_regrassion(features, target))

del noise, noises
