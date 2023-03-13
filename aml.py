"""Helper functions for Exercise Class 1: Fundamentals of Machine Learning in Python.

Includes functions:
    plot_validation_curve
        (used in exercise B, task 4)
    compare_models_cross_validation
        (used in exercise C, task 2)

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate


def plot_validation_curve(train_scores,
                          test_scores,
                          param_range,
                          title='Validation Curve',
                          xlabel='Hyperparameter',
                          linewidth=2):
    """Plot validation curve from train and test scores obtained from sklearn.model_selection.validation_curve.

    Parameters
    ----------
    train_scores : ndarray
        Output from sklearn.model_selection.validation_curve.
    test_scores : ndarray
        Output from sklearn.model_selection.validation_curve.
    param_range : iterable
        Hyperparameter values.
    title : str, default 'Validation Curve'
        Plot title, should contain model name.
    xlabel : str, default 'Hyperparameter'
        Label fo x-axis, should contain hyperparameter name.
    linewidth: int, default 2
        Width of plot lines.

    Returns
    -------
    None
    """

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set(ylim=(0.0, 1.1))

    ax.plot(
        param_range, train_scores_mean, label="Training score", color="darkorange", lw=linewidth)
    plt.fill_between(
        param_range,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2,
        color="darkorange",
        lw=linewidth)

    ax.plot(
        param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=linewidth)
    plt.fill_between(
        param_range,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.2,
        color="navy",
        lw=linewidth)

    plt.legend(loc="best")
    plt.show()


def compare_models_cross_validation(X, y, model_names=None):
    """Compare cross-validated NMAPE, NMSE, and R2 for different models.

    Parameters
    ----------
    X : dataframe
        Features.
    y : series
        Target variable.
    model_names : list of str, optional
        Names of models to be evaluated. Possible names:
        'k_neighbors_regressor', 'linear_regression', 'random_forrest_regressor', 'SVR'
        (defaults to all of these models)

    Returns
    -------
    dataframe
        Indexed by chosen models, columns are metrics NMAPE, NMSE, R2.
    """

    models = {
        'k_neighbors_regressor': KNeighborsRegressor(),
        'linear_regression': LinearRegression(),
        'random_forrest_regressor': RandomForestRegressor(),
        'SVR': SVR()
    }

    if model_names is None:
        model_names = models.keys()

    rows = []

    for name in model_names:

        m = models[name]
        m.fit(X, y)

        cv = cross_validate(m, X, y, scoring=('neg_mean_absolute_percentage_error',
                                              'neg_mean_squared_error',
                                              'r2'))

        nmape = np.mean(cv.get('test_neg_mean_absolute_percentage_error'))
        nmse = np.mean(cv.get('test_neg_mean_squared_error'))
        r2 = np.mean(cv.get('test_r2'))

        rows.append((name, nmape, nmse, r2))

    comparison = pd.DataFrame(rows, columns=[['model', 'NMAPE', 'NMSE', 'R2']]).set_index('model')

    return comparison
