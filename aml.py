"""This module contains helper functions for code in the aml repository.

Includes functions:
    fancy_print
    plot_validation_curve
    plot_grid_search_results
    compare_models_cross_validation
    features_by_importance

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import dummy, ensemble, linear_model, naive_bayes, neighbors, svm, tree
from sklearn.model_selection import cross_validate


def fancy_print(label, value):
    """Print labeled values in a crisp, sexy fashion.

    Parameters
    ----------
    label : str
        Label of value to print.
    value : printable
        Value to print.

    Returns
    -------
    None
    """
    print(f'{label + " ":.<40}', value)


def plot_validation_curve(
        train_scores, test_scores, param_range, title='Validation Curve', xlabel='Hyperparameter', linewidth=2):
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


def plot_grid_search_results(grid_search, hyp_par_1, hyp_par_2):
    """Plot heatmap of prediction scores over a 2D hyperparameter grid.

    Parameters
    ----------
    grid_search : sklearn.model_selection.GridSearchCV
        GridSearchCV instance which was used to perform the grid search.
    hyp_par_1 : str
        Name of hyperparameter to be shown on the y-axis.
    hyp_par_2 : str
        Name of hyperparameter to be shown on the x-axis.

    Returns
    -------
    None
    """

    hp1 = grid_search.param_grid[hyp_par_1]
    hp2 = grid_search.param_grid[hyp_par_2]
    score = grid_search.cv_results_['mean_test_score'].reshape(len(hp1), len(hp2))

    plt.figure(figsize=(6, 4), dpi=150)
    plt.title('Validation accuracy')
    plt.imshow(score)
    plt.yticks(np.arange(len(hp1)), hp1)
    plt.xticks(np.arange(len(hp2)), hp2, rotation=45)
    plt.ylabel(hyp_par_1)
    plt.xlabel(hyp_par_2)
    plt.colorbar()
    plt.show()


def compare_models_cross_validation(X, y, which='regression', model_names=None, scoring=None, hyperparameters=None):
    """Compare cross-validated metrics for different models.

    Parameters
    ----------
    X : DataFrame
        Features.
    y : Series
        Target variable.
    which: {'regression', 'classification'}, optional
        Type of models to be evaluated.
        (defaults to 'regression')
    model_names : list of str, optional
        Names of models to be evaluated.
        (defaults to all models of type `which`)
    scoring : list of str, optional
        Names of metrics to evaluate models in.
        (defaults to all metrics for type `which`)
    hyperparameters : dict of str: dict, optional
        Keys are model names, values are dictionaries. The keys of those inner dictionaries are hyperparameter names
        for that model, while their values are the corresponding hyperparameter values.
        (defaults to default hyperparameter values for each model)

    Returns
    -------
    DataFrame
        Indexed by chosen models, columns are chosen metrics.
    """

    models = {
        'regression': {
            'decision_tree_regressor': tree.DecisionTreeRegressor,
            'dummy_regressor': dummy.DummyRegressor,
            'k_neighbors_regressor': neighbors.KNeighborsRegressor,
            'linear_regression': linear_model.LinearRegression,
            'random_forest_regressor': ensemble.RandomForestRegressor,
            'SVR': svm.SVR
        },
        'classification': {
            'decision_tree_classifier': tree.DecisionTreeClassifier,
            'dummy_classifier': dummy.DummyClassifier,
            'gaussian_nb': naive_bayes.GaussianNB,
            'k_neighbors_classifier': neighbors.KNeighborsClassifier,
            'random_forest_classifier': ensemble.RandomForestClassifier,
        }
    }

    metrics = {
        'regression': ['neg_mean_absolute_percentage_error',
                       'neg_mean_squared_error',
                       'r2'],
        'classification': ['accuracy',
                           'f1_micro',
                           'f1_macro',
                           'roc_auc']
    }

    if model_names is None:
        model_names = models.get(which).keys()

    if scoring is None:
        scoring = metrics.get(which)

    if hyperparameters is None:
        hyperparameters = {}

    rows = []
    for mn in model_names:

        if mn not in hyperparameters.keys():
            hyperparameters[mn] = {}

        m = models.get(which).get(mn)(**hyperparameters.get(mn))
        m.fit(X, y)

        cv = cross_validate(m, X, y, scoring=tuple(scoring))

        r = [mn]
        for s in scoring:
            r.append(np.mean(cv[f'test_{s}']))

        rows.append(r)

    columns = [['model'] + scoring]
    comparison = pd.DataFrame(rows, columns=columns).set_index('model')

    return comparison


def features_by_importance(X, y, n=5, model='random_forest_classifier', random_state=None, plot=False):
    """Return list of most important features.

    Parameters
    ----------
    X : DataFrame
        Features.
    y : Series
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
