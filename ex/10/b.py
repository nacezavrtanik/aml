"""Class 10, Exercise B: Equation Discovery with the BACON algorithm"""

import numpy as np
import pandas as pd

import data_generators
from aml import fancy_print


LAW = "Newton's Second Law"
DATA_GENERATOR = data_generators.generate_newton


# 1 Implement the BACON algorithm
def bacon(df, max_steps=20, epsilon=1e-12):

    step = 0
    stds = np.std(df)
    columns = df.columns

    while step < max_steps and min(stds) > epsilon:

        correlations = np.corrcoef(df.rank().T)
        n = correlations.shape[0]
        correlations = [(abs(correlations[i, j]), i, j, correlations[i, j] < 0)
                        for i in range(n)
                        for j in range(i+1, n)]
        correlations = sorted(correlations, reverse=True)

        for _, i, j, is_multiplication in correlations:

            name_i, name_j = columns[i], columns[j]

            if is_multiplication:
                column = df[name_i] * df[name_j]
                column.name = f"({name_i}) * ({name_j})"
            else:
                column = df[name_i] / df[name_j]
                column.name = f"({name_i}) / ({name_j})"

            if column.name not in columns:
                df = pd.concat([df, column], axis=1)
                stds = np.std(df)
                columns = df.columns
                break

        step += 1

    constant_column = columns[np.argmin(stds)]
    constant = np.mean(df[constant_column])

    equation = f"Equation: {constant:.2f} = {constant_column}\nSteps: {step}\nError: {min(stds):.4e}"

    return equation


# 2 Test the BACON algorithm on given data
fancy_print(f"BACON Algorithm for {LAW}")
print(bacon(DATA_GENERATOR(100, noise=0)))
del LAW
