"""This module contains helper functions for code in the aml repository.

Includes functions:
    fancy_print
    plot_validation_curve
    plot_grid_search_results
    compare_models_cross_validation
    features_by_importance

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import express as px
from sklearn import dummy, ensemble, linear_model, naive_bayes, neighbors, svm, tree
from sklearn.model_selection import cross_validate


def fancy_print(*args, parse_dict=True):
    """Print stuff in a crisp, sexy fashion.

    Parameters
    ----------
    *args
        One or two positional arguments to be printed, fancily.

        If a single positional argument is passed, it will be printed as a
        title (i.e. underlined), unless it is a dictionary and `parse_dict`
        is set to True, in which case separate lines of key-value pairs will
        be printed.

        If two positional arguments are passed, they will be printed in the
        same line, separated by dots.

    parse_dict : bool, default True
        If True, and only a single positional argument is passed, and it is a
        dictionary, key-value pairs will be printed on separate lines, fancily.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If a positional argument has no `__str__` method.
    TypeError
        If the number of passed positional arguments is not 1 or 2.

    Examples
    --------
    >>> fancy_print('Very nice title')

    Very nice title
    ---------------

    >>> fancy_print('Number', 13)
    Number ................................. 13

    >>> my_dict = {123: 'Potato',
    ...            'pi': 3.14,
    ...            'jobs': {'Alice': 'Farmer', 'Bob': 'Fishmonger'}}
    >>> fancy_print(my_dict, parse_dict=True)
    123 .................................... Potato
    pi ..................................... 3.14
    jobs ................................... {'Alice': 'Farmer', 'Bob': 'Fishmonger'}
    """

    for argument in args:
        if '__str__' not in dir(argument):
            raise TypeError(f"no method '__str__' for instance '{argument}'")

    if len(args) == 1:

        argument, = args
        if isinstance(argument, dict) and parse_dict:
            for key, value in argument.items():
                fancy_print(key, value)
        else:
            title_length = len(str(argument))
            print(f"\n{argument}\n{title_length * '-'}")

    elif len(args) == 2:

        argument_1, argument_2 = args
        print(f"{str(argument_1) + ' ':.<40}", argument_2)

    else:
        raise TypeError(f'expected 1 or 2 positional arguments, got {len(args)} instead')


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


def _fill_missing_values(x):
    """Apply on series: fill missing values with `numpy.nan`.

    Helper function for `plot_algorithm_and_hyperparameter_comparison`.
    """
    if x:
        return x[0]
    return np.nan


def _algorithm_name_from_int(n, algorithm_names):
    """Apply on dataframe: replace integers with appropriate name strings.

    Helper function for `plot_algorithm_and_hyperparameter_comparison`.
    """
    for i, algorithm_name in enumerate(algorithm_names):
        if n == i:
            return algorithm_name


def _add_hover_data(fig, df, algorithm_name):
    """On hover, display hyperparameters for specified algorithm.

    Helper function for `plot_algorithm_and_hyperparameter_comparison`.
    """

    df_for_algorithm = df[df['algorithm'] == algorithm_name].dropna(axis=1)
    columns = df_for_algorithm.columns

    customdata = df.loc[df_for_algorithm.index, columns]
    hovertemplate = '<br>'.join([f'{column}: %{{customdata[{i}]}}' for i, column in enumerate(columns)])
    selector = {'name': algorithm_name}

    fig.update_traces(customdata=customdata, hovertemplate=hovertemplate, selector=selector)
    return fig


def plot_algorithm_and_hyperparameter_comparison(trials, algorithm_names, label='algorithm'):
    """Plot results of optimisation with the `hyperopt` library.

    Parameters
    ----------
    trials : hyperopt.Trials
        After concluded optimisation, already containing trial data.
    algorithm_names : list of str
        Names of algorithms. These are required to be the same names used in
        the `space` kwarg of `hyperopt.fmin`, listed in the same order.
    label : str, default 'algorithm'
        Name of the key used in the `space` dictionary to represent the
        compared algorithms.

    Returns
    -------
    None
    """

    trials_df = pd.DataFrame([pd.Series(t['misc']['vals']).apply(_fill_missing_values) for t in trials])
    trials_df['loss'] = [t['result']['loss'] for t in trials]
    trials_df['trial_number'] = trials_df.index
    trials_df[label] = trials_df[label].apply(lambda x: _algorithm_name_from_int(x, algorithm_names))

    fig = px.scatter(trials_df, x='trial_number', y='loss', color=label)
    for algorithm_name in algorithm_names:
        fig = _add_hover_data(fig, trials_df, algorithm_name)

    fig.show()


def compare_models_cross_validation(X, y, which='regression', model_names=None, scoring=None, hyperparameters=None):
    """Compare cross-validated metrics for different models.

    Parameters
    ----------
    X : pandas.DataFrame
        Features.
    y : pandas.Series
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
    hyperparameters : dict, optional
        Keys are model names, values are dictionaries. The keys of those inner dictionaries are hyperparameter names
        for that model, while their values are the corresponding hyperparameter values.
        (defaults to default hyperparameter values for each model)

    Returns
    -------
    pandas.DataFrame
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
    X : pandas.DataFrame
        Features.
    y : pandas.Series
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
