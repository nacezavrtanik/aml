"""Class 4, Exercise A: Hyperparameter Optimisation with hyperopt"""

import numpy as np
from hyperopt import hp, tpe, rand, fmin, Trials, space_eval, pyll, base
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from aml import fancy_print


MAX_EVALS = 100

# 1 Minimise function of one variable, uniform distribution
space_1 = {'x': hp.uniform('x', 4, 8)}


def objective_1(params):
    """Function to be minimised."""
    x = params.get('x')
    return x**2


trials_1 = Trials()
best_1 = fmin(fn=objective_1,
              space=space_1,
              algo=tpe.suggest,
              max_evals=MAX_EVALS,
              trials=trials_1)

best_1 = space_eval(space_1, best_1)
fancy_print('Best x', best_1.get('x'))
fancy_print('Best value', objective_1(best_1))

# 2 Minimise function of two variables, normal distribution
space_2 = {'x1': hp.normal('x1', -2, 2),
           'x2': hp.normal('x2', -1, 3)}


def objective_2(params):
    """Function to be minimised."""
    x1 = params.get('x1')
    x2 = params.get('x2')
    return (x1 - x2)**2 + (x2 - 1)**2


trials_2 = Trials()
best_2 = fmin(fn=objective_2,
              space=space_2,
              algo=tpe.suggest,
              max_evals=MAX_EVALS,
              trials=trials_2)

best_2 = space_eval(space_2, best_2)
fancy_print('Best x1', best_2.get('x1'))
fancy_print('Best x2', best_2.get('x2'))
fancy_print('Best value', objective_2(best_2))


# 3 Find best algorithm
n_cases = 1000
n_features = 5

np.random.seed(0)
X = np.random.rand(n_cases, n_features)
y = np.dot(X, list(range(n_features)))

average = np.mean(y)
positive = y >= average
y[positive] = 1
y[~positive] = 0

space_3 = {'algorithm_name': hp.choice('algorithm', ['k_neighbors_classifier', 'decision_tree_classifier', 'svc'])}

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


def objective_3(params):
    """Function to be minimised."""

    algorithm_name = params.get('algorithm_name')
    if algorithm_name == 'k_neighbors_classifier':
        algorithm = KNeighborsClassifier()
    elif algorithm_name == 'decision_tree_classifier':
        algorithm = DecisionTreeClassifier()
    elif algorithm_name == 'svc':
        algorithm = SVC()
    else:
        raise ValueError(f'Invalid parameter \'algorithm_name\': {algorithm_name}')

    algorithm.fit(X_train, y_train)
    score = algorithm.score(X_test, y_test)
    return 1 - score


trials_3 = Trials()
best_3 = fmin(fn=objective_3,
              space=space_3,
              algo=tpe.suggest,
              max_evals=MAX_EVALS,
              trials=trials_3)

best_3 = space_eval(space_3, best_3)
fancy_print('Best algorithm', best_3.get('algorithm_name'))
fancy_print('Best score', 1 - objective_3(best_3))

# 4 Define hyperparameter space
space_4 = {
    'algorithm': hp.choice('algorithm', [
        {
            'name': 'decision_tree_classifier',
            'max_depth': hp.choice('max_depth', [2, 4, 8, 16, 32]),
            'n_estimators': hp.choice('n_estimators', [2, 5, 10, 20, 50, 100, 150, 200, 500])
        },
        {
            'name': 'k_neighbors_classifier',
            'n_neighbors': hp.choice('n_neighbors', [1, 2, 5, 10, 20, 50])
        },
        {
            'name': 'svm',
            'C': hp.lognormal('C', 0, 1),
            'kernel': hp.choice('kernel', [
                {
                    'type': 'linear'
                },
                {
                    'type': 'rbf',
                    'gamma': hp.lognormal('gamma', 0, 1)
                },
                {
                    'type': 'poly',
                    'degree': hp.choice('degree', [1, 2, 3, 5, 10, 20])
                }
            ]),
        },
    ])
}
