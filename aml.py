"""This module contains helper functions for code in the aml repository.

Includes functions:
    plot_validation_curve
        (used in: ex1b1)
    compare_models_cross_validation
        (used in: ex1c2, ex3a2)
    features_by_importance
        (used in: ex3a3)

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import dummy, ensemble, linear_model, naive_bayes, neighbors, svm, tree
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


def compare_models_cross_validation(X, y, which, model_names):
    """Compare cross-validated metrics for different models.

    Parameters
    ----------
    X : dataframe
        Features.
    y : series
        Target variable.
    which: {'regression', 'classification'}
        Type of models to be evaluated.
    model_names : list of str
        Names of models to be evaluated. Possible names: keys of dictionary `models` below.

    Returns
    -------
    dataframe
        Indexed by chosen models, columns are metrics NMAPE, NMSE, R2.
    """

    # TODO `which` defaults to 'regression'
    # TODO `model_names` defaults to all models
    # TODO add option to pass hyperparameters
    # TODO add parameter `metrics`, defaults to all metrics for that `which`
    # TODO generalize repeated code
    # TODO adjust docstring, adhere to PEP

    models = {
        'decision_tree_classifier': tree.DecisionTreeClassifier(),
        'decision_tree_regressor': tree.DecisionTreeRegressor(),
        'dummy_classifier': dummy.DummyClassifier(),
        'dummy_regressor': dummy.DummyRegressor(),
        'gaussian_nb': naive_bayes.GaussianNB(),
        'k_neighbors_classifier': neighbors.KNeighborsClassifier(),
        'k_neighbors_regressor': neighbors.KNeighborsRegressor(),
        'linear_regression': linear_model.LinearRegression(),
        'random_forest_classifier': ensemble.RandomForestClassifier(),
        'random_forest_regressor': ensemble.RandomForestRegressor(),
        'SVR': svm.SVR()
    }

    rows = []

    for name in model_names:

        m = models[name]
        m.fit(X, y)

        if which == 'regression':
            cv = cross_validate(m, X, y, scoring=('neg_mean_absolute_percentage_error',
                                                  'neg_mean_squared_error',
                                                  'r2'))

            nmape = np.mean(cv.get('test_neg_mean_absolute_percentage_error'))
            nmse = np.mean(cv.get('test_neg_mean_squared_error'))
            r2 = np.mean(cv.get('test_r2'))

            rows.append((name, nmape, nmse, r2))

            comparison = pd.DataFrame(rows, columns=[['model', 'NMAPE', 'NMSE', 'R2']]).set_index('model')

        elif which == 'classification':
            cv = cross_validate(m, X, y, scoring=('accuracy', 'f1_micro', 'f1_macro'))

            accuracy = np.mean(cv.get('test_accuracy'))
            micro = np.mean(cv.get('test_f1_micro'))
            macro = np.mean(cv.get('test_f1_macro'))

            rows.append((name, accuracy, micro, macro))

            comparison = pd.DataFrame(rows, columns=[['model', 'accuracy', 'f1_micro', 'f1_macro']]).set_index('model')

    return comparison


def features_by_importance(X, y, n=5, model='random_forest_classifier', random_state=None, plot=False):
    """Return list of most important features.

    Parameters
    ----------
    X : dataframe
        Features.
    y : series
        Target variable.
    n : int
        Number of features to take.
    model : {'decision_tree_classifier', 'random_forest_classifier'}, optional
        Model to base importance on.
        (defaults to 'random_forest_classifier')
    random_state : int, optional
        Random seed (set for repeatable function calls).
        (defaults to None)
    plot : bool, optional
        If True, a bar plot of features by importance is displayed.
        (defaults to False)

    Returns
    -------
    list of str
        The `n` most important features.
    """

    models = {'decision_tree_classifier': tree.DecisionTreeClassifier(random_state=random_state),
              'random_forest_classifier': ensemble.RandomForestClassifier(random_state=random_state)
              }

    m = models.get(model)
    m.fit(X, y)

    # Borrowed code
    features = X.columns
    importances = m.feature_importances_
    sorted_ids = np.argsort(-importances)
    sorted_importances = importances[sorted_ids]
    sorted_features = features[sorted_ids]

    names = list(sorted_features[:n])

    if plot:
        # Borrowed code
        xs = range(len(features))
        plt.figure()
        plt.bar(xs, sorted_importances)
        plt.xticks(xs, sorted_features, rotation='vertical')
        plt.ylabel('Importance')
        plt.title('Features by Importance')
        plt.tight_layout()
        plt.show()

    return names
