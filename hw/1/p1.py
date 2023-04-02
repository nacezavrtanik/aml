"""Assignment 1, Problem 1: Method Selection and Hyperparameter Optimisation"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from hyperopt import Trials, fmin, tpe, space_eval, hp

from aml import compare_models_cross_validation, fancy_print


np.random.seed(0)

TYPE = 'classification'
METRIC = 'roc_auc'

data = pd.read_csv('podatki.csv')
X, y = data.drop(columns='y'), data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


# 1 Manual approach


# 1.1 Select model

comparison_models = compare_models_cross_validation(X, y, which=TYPE, scoring=[METRIC])

# 1.2 Select hyperparameters

N_ESTIMATORS = range(4, 151, 24)  # default is 100 (is included)
MAX_DEPTH = list(range(2, 51, 6)) + [None]  # default is None
MIN_SAMPLES_SPLIT = range(2, 500, 50)  # default is 2


def calculate_roc_auc_for_random_forest(hyperparameters, print_title=None):
    """Fit RandomForestClassifier with train data and evaluate on test data.

    Parameters
    ----------
    hyperparameters : dict
        Keys are hyperparameter names, values are hyperparameter values.
    print_title : str, optional
        If provided, `print_title` will be printed, followed by hyperparameter values and the roc_auc score.
        (defaults to None)

    Returns
    -------
    float
        The roc_auc score.
    """

    model = RandomForestClassifier(**hyperparameters)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_score)

    if print_title:
        print(print_title)
        for k, v in hyperparameters.items():
            fancy_print(k, v)
        fancy_print(METRIC, roc_score)
        print('')

    return roc_score


# 1.2.1 Completely manual

hyperparams = [{'random_forest_classifier': {'n_estimators': i,
                                             'max_depth': j,
                                             'min_samples_split': k}}
               for i in N_ESTIMATORS
               for j in MAX_DEPTH
               for k in MIN_SAMPLES_SPLIT]

if input('Start manual hyperparameter optimisation? (y/n) ').lower() == 'y':

    print('Starting manual hyperparameter optimisation ...')
    rows = []
    for h in hyperparams:

        scores = compare_models_cross_validation(X_train, y_train,
                                                 which=TYPE,
                                                 model_names=['random_forest_classifier'],
                                                 hyperparameters=h,
                                                 scoring=[METRIC])
        s = scores.iloc[0, 0]
        inner = h.get('random_forest_classifier')
        r = (inner.get('n_estimators'), inner.get('max_depth'), inner.get('min_samples_split'), s)
        rows.append(r)

    comparison_hyperparams = pd.DataFrame(rows, columns=['n_estimators', 'max_depth', 'min_samples_split', METRIC]
                                          ).sort_values(by=METRIC, ascending=False)

    best_hyperparams_manual = {'n_estimators': 50,
                               'max_depth': 32,
                               'min_samples_split': 5}  # from `comparison_hyperparams`
    roc_score_manual = calculate_roc_auc_for_random_forest(
        best_hyperparams_manual, print_title='HYPERPARAMETER OPTIMISATION -- Manual')

    del rows, h, scores, s, inner, r
del hyperparams


# 1.2.2 With GridSearchCV

hyperparams = {'n_estimators': list(N_ESTIMATORS),
               'max_depth': list(MAX_DEPTH),
               'min_samples_split': list(MIN_SAMPLES_SPLIT)}

if input('Start grid search? (y/n) ').lower() == 'y':

    print('Starting grid search ...')
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    gs = GridSearchCV(rfc, param_grid=hyperparams, scoring=METRIC)
    gs.fit(X_train, y_train)

    best_hyperparams_GridSearchCV = gs.best_params_
    roc_score_GridSearchCV = calculate_roc_auc_for_random_forest(
        best_hyperparams_GridSearchCV, print_title='HYPERPARAMETER OPTIMISATION -- GridSeachCV')

    del rfc, gs
del hyperparams


# 2 Automated approach


# 2.1 Select model, define hyperparameter space
hyperparams = {
    "algo": hp.choice('algo', [
        {
            'name': 'random_forest_classifier',
            'n_estimators': hp.choice('n_estimators', N_ESTIMATORS),
            'max_depth_forest': hp.choice('max_depth_forest', MAX_DEPTH),
            'min_samples_split': hp.choice('min_samples_split', MIN_SAMPLES_SPLIT)
        },
        {
            'name': 'k_neighbors_classifier',
            'n_neighbors': hp.choice("n_neighbors", [1, 2, 3, 4, 5, 10, 15, 50, 100])
        },
        {
            'name': 'decision_tree_classifier',
            'max_depth_tree': hp.choice('max_depth_tree', MAX_DEPTH)
        },
    ])
}


def criterion_function(parameters):
    """Function to minimize with hyperopt."""

    algorithm = parameters.get('algo')
    algorithm_name = algorithm.get('name')

    if algorithm_name == 'random_forest_classifier':
        model = RandomForestClassifier(n_estimators=algorithm.get('n_estimators'),
                                       max_depth=algorithm.get('max_depth_forest'),
                                       min_samples_split=algorithm.get('min_samples_split'))
    elif algorithm_name == 'decision_tree_classifier':
        model = DecisionTreeClassifier(max_depth=algorithm.get('max_depth_tree'))
    elif algorithm_name == 'k_neighbors_classifier':
        model = KNeighborsClassifier(n_neighbors=algorithm.get('n_neighbors'))
    else:
        raise ValueError('Illegitimate value for parameter \'algorithm_name\'!')

    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    roc_score = roc_auc_score(y_test, y_score)

    return 1 - roc_score


if input('Start automated hyperparameter optimisation? (y/n) ').lower() == 'y':

    print('Starting automated hyperparameter optimisation ...')
    trials = Trials()
    best = fmin(fn=criterion_function,
                space=hyperparams,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    best_hyperparams_hyperopt = space_eval(hyperparams, best)
    roc_score_hyperopt = 1 - criterion_function(best_hyperparams_hyperopt)
    print('HYPERPARAMETER OPTIMISATION -- hyperopt')
    fancy_print(METRIC, roc_score_hyperopt)

    del trials, best
del hyperparams

del N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT
