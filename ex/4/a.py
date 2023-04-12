"""Class 4, Exercise A: Hyperparameter Optimisation with hyperopt"""

import numpy as np
from hyperopt import hp, tpe, fmin, Trials, space_eval
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from aml import fancy_print


MAX_EVALS = 100

# 1 Minimise function of one variable, uniform distribution
fancy_print('TASK 1')
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
fancy_print('TASK 2')

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
fancy_print('TASK 3')

K_NEIGHBORS_CLASSIFIER = 'k_neighbors_classifier'
DECISION_TREE_CLASSIFIER = 'decision_tree_classifier'
SUPPORT_VECTOR_CLASSIFIER = 'support_vector_classifier'

NUMBER_OF_CASES = 1000
NUMBER_OF_FEATURES = 5

np.random.seed(0)
X = np.random.rand(NUMBER_OF_CASES, NUMBER_OF_FEATURES)
y = np.dot(X, list(range(NUMBER_OF_FEATURES)))

average = np.mean(y)
positive = y >= average
y[positive] = 1
y[~positive] = 0

space_3 = {'algorithm_name': hp.choice('algorithm', [K_NEIGHBORS_CLASSIFIER,
                                                     DECISION_TREE_CLASSIFIER,
                                                     SUPPORT_VECTOR_CLASSIFIER])}

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y, train_size=0.8)


def objective_3(params):
    """Function to be minimised."""

    algorithm_name = params.get('algorithm_name')
    if algorithm_name == K_NEIGHBORS_CLASSIFIER:
        algorithm = KNeighborsClassifier()
    elif algorithm_name == DECISION_TREE_CLASSIFIER:
        algorithm = DecisionTreeClassifier()
    elif algorithm_name == SUPPORT_VECTOR_CLASSIFIER:
        algorithm = SVC()
    else:
        raise ValueError(f'Invalid parameter \'algorithm_name\': {algorithm_name}')

    algorithm.fit(X_train_3, y_train_3)
    score = algorithm.score(X_test_3, y_test_3)
    return 1 - score


trials_3 = Trials()
best_3 = fmin(fn=objective_3,
              space=space_3,
              algo=tpe.suggest,
              max_evals=MAX_EVALS,
              trials=trials_3)

best_3 = space_eval(space_3, best_3)
fancy_print('Best algorithm', best_3.get('algorithm_name'))
fancy_print('Score', 1 - objective_3(best_3))
del X, y, average, positive, X_train_3, X_test_3, y_train_3, y_test_3

# 4 Define hyperparameter space
space_4 = {
    'algorithm': hp.choice('algorithm', [
        {
            'name': DECISION_TREE_CLASSIFIER,
            'max_depth': hp.choice('max_depth', [2, 4, 8, 16, 32]),
            'min_samples_split': hp.choice('min_samples_split', [2, 5, 10, 20, 50])
        },
        {
            'name': K_NEIGHBORS_CLASSIFIER,
            'n_neighbors': hp.choice('n_neighbors', [1, 2, 5, 10, 20, 50])
        },
        {
            'name': SUPPORT_VECTOR_CLASSIFIER,
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

# 5 Compare algorithm performance with default and optimised hyperparameters
fancy_print('TASK 5')


def set_target_variable(xs, _type):
    """Calculate target variabe from given data.

    This function is used to create the target variable for randomly generated input datasets. Together, they are used
    as dummy datasets for testing optimisation methods.

    Parameters
    ----------
    xs : numpy.ndarray
        Dataset containing features only (no target variable).
    _type : {1, 2, 3}
        Method for generating the target variable.

    Returns
    -------
    numpy.ndarray
        Array of values for the traget variable.

    Raises
    ------
    ValueError
        If `_type` not in {1, 2, 3}.
    """
    if _type == 1:
        r = list(range(xs.shape[1]))
        cs = np.array(r).reshape((-1, 1))
        ys = np.dot(xs, cs).reshape((-1,))
        is_positive = ys >= np.mean(ys)
        ys = np.array(['a' if p else 'b' for p in is_positive])

    elif _type == 2:
        group1 = xs[:, 1] > 0.5
        group2 = (xs[:, 1] <= 0.5) & (xs[:, 2] > 0.2)
        group3 = (xs[:, 1] <= 0.5) & (xs[:, 2] <= 0.2)
        ys = np.zeros(xs.shape[0], dtype=str)
        ys[group1] = 'a'
        ys[group2] = 'b'
        ys[group3] = 'c'

    elif _type == 3:
        circle = np.sum(np.square(xs), axis=1) > 0.4
        ys = np.zeros(xs.shape[0], dtype=str)
        ys[circle] = 'a'
        ys[~circle] = 'b'

    else:
        raise ValueError(f'Invalid parameter \'_type\': {_type}')

    return ys


def prepare_dataset(n_cases, n_features, _type, randomisation_frequency=0.2, train_size=0.8):
    """Create random dataset.

    Parameters
    ----------
    n_cases : int
        Number of cases.
    n_features : int
        Number of features.
    _type : {1, 2, 3}
        Method for generating the target variable. Passed into function `set_target_variable`.
    randomisation_frequency : float, optional
        Likelyhood of changing each value of the target variable.
        Values between 0 and 1. Value 0 indicates no target values will be modified. Value 1 indicates
        all target values will be modifed.
        (defaults to 0.2)
    train_size : float, int, optional
        Size of train data. Passed into function `sklearn.model_selection.train_test_split`.
        Either between 0 and 1 (if float) or between 0 and `n_cases` (if int).
        (defaults to 0.8)

    Returns
    -------
    x_train : numpy.ndarray
        Train data.
    x_test : numpy.ndarray
        Test data.
    target_train : numpy.ndarray
        Train target variable.
    target_test : numpy.ndarray
        Test target variable.
    """
    x = np.random.rand(n_cases, n_features)
    target = set_target_variable(x, _type)
    possible_target_values = list(set(target))
    for i in range(len(target)):
        if np.random.rand(1) < randomisation_frequency:
            i0 = possible_target_values.index(target[i])
            target[i] = possible_target_values[(i0 + 1) % len(possible_target_values)]
    x_train, x_test, target_train, target_test = train_test_split(x, target, train_size=train_size)
    return x_train, x_test, target_train, target_test


space_5 = space_3
space_5_optimised = space_4
del space_4

for t in [1, 2, 3]:

    X_train_5, X_test_5, y_train_5, y_test_5 = prepare_dataset(NUMBER_OF_CASES, NUMBER_OF_FEATURES, t)

    def objective_5(params):
        """Function to be minimised."""

        algorithm_name = params.get('algorithm_name')
        if algorithm_name == K_NEIGHBORS_CLASSIFIER:
            algorithm = KNeighborsClassifier()
        elif algorithm_name == DECISION_TREE_CLASSIFIER:
            algorithm = DecisionTreeClassifier()
        elif algorithm_name == SUPPORT_VECTOR_CLASSIFIER:
            algorithm = SVC()
        else:
            raise ValueError(f'Invalid parameter \'algorithm_name\': {algorithm_name}')

        algorithm.fit(X_train_5, y_train_5)
        score = algorithm.score(X_test_5, y_test_5)
        return 1 - score

    def objective_5_optimised(params):
        """Function to be minimised."""

        algorithm = params.get('algorithm')
        algorithm_name = algorithm.get('name')

        if algorithm_name == K_NEIGHBORS_CLASSIFIER:
            algorithm_temp = KNeighborsClassifier(n_neighbors=algorithm.get('n_neighbors'))

        elif algorithm_name == DECISION_TREE_CLASSIFIER:
            algorithm_temp = DecisionTreeClassifier(max_depth=algorithm.get('max_depth'),
                                                    min_samples_split=algorithm.get('min_samples_split'))

        elif algorithm_name == SUPPORT_VECTOR_CLASSIFIER:

            c = algorithm.get('C')
            kernel = algorithm.get('kernel').get('type')

            dummy_value = 1
            if kernel == "linear":
                degree = dummy_value
                gamma = dummy_value
            elif kernel == "rbf":
                gamma = algorithm.get('kernel').get('gamma')
                degree = dummy_value
            elif kernel == 'poly':
                gamma = dummy_value
                degree = algorithm.get('kernel').get('degree')
            else:
                raise ValueError(f'Invalid value for \'kernel\': {kernel}')

            algorithm_temp = SVC(kernel=kernel, gamma=gamma, C=c, degree=degree)

        else:
            raise ValueError(f'Invalid value for \'algorithm_name\': {algorithm_name}')

        algorithm = algorithm_temp
        algorithm.fit(X_train_5, y_train_5)
        score = algorithm.score(X_test_5, y_test_5)
        return 1 - score

    fancy_print(f'DEFAULT HYPERPARAMETERS -- Data type {t}')
    trials_5 = Trials()
    best_5 = fmin(fn=objective_5,
                  space=space_5,
                  algo=tpe.suggest,
                  max_evals=MAX_EVALS,
                  trials=trials_5)

    best_5 = space_eval(space_5, best_5)
    fancy_print('Best algorithm', best_5.get('algorithm_name'))
    fancy_print('Score', 1 - objective_5(best_5))

    fancy_print(f'OPTIMISED HYPERPARAMETERS -- Data type {t}')
    trials_5_optimised = Trials()
    best_5_optimised = fmin(fn=objective_5_optimised,
                            space=space_5_optimised,
                            algo=tpe.suggest,
                            max_evals=MAX_EVALS,
                            trials=trials_5_optimised)

    best_5_optimised = space_eval(space_5_optimised, best_5_optimised)
    fancy_print(best_5_optimised.get('algorithm'))
    fancy_print('Score', 1 - objective_5_optimised(best_5_optimised))
    del t, X_train_5, X_test_5, y_train_5, y_test_5

del KNeighborsClassifier, DecisionTreeClassifier, SVC
del K_NEIGHBORS_CLASSIFIER, DECISION_TREE_CLASSIFIER, SUPPORT_VECTOR_CLASSIFIER
