"""Assignment 1, Problem 1: Method Selection and Hyperparameter Optimisation"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

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

if __name__ == '__main__':
    print(' Starting manual hyperparameter optimisation ...')
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

if __name__ == '__main__':
    print(' Starting grid search ...')
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    gs = GridSearchCV(rfc, param_grid=hyperparams, scoring=METRIC)
    gs.fit(X_train, y_train)

    best_hyperparams_GridSearchCV = gs.best_params_
    roc_score_GridSearchCV = calculate_roc_auc_for_random_forest(
        best_hyperparams_GridSearchCV, print_title='HYPERPARAMETER OPTIMISATION -- GridSeachCV')

    del rfc, gs
del hyperparams

del N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT
