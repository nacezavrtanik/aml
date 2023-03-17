"""Class 3, Exercise B: Hyperparameter Optimisation"""

import pandas as pd

from aml import compare_models_cross_validation


# 1 Vary hyperparameters in a decision tree model
data = pd.read_csv('drugi_del_podatki.csv')
X, y = data.drop('y', axis=1), data['y']

hyperparams = [{'decision_tree_regressor': {'max_depth': i,
                                            'min_samples_split': j}}
               for i in [2, 4, 8, 32, 128]
               for j in [5, 10, 50, 100, 500]]

rows = []
for h in hyperparams:

    scores = compare_models_cross_validation(X, y,
                                             model_names=['decision_tree_regressor'],
                                             hyperparameters=h,
                                             scoring=['r2'])
    s = scores.iloc[0, 0]
    inner = h.get('decision_tree_regressor')
    r = (inner.get('max_depth'), inner.get('min_samples_split'), s)
    rows.append(r)

r2 = pd.DataFrame(rows, columns=['max_depth', 'min_samples_split', 'r2']).sort_values(by='r2')
del hyperparams, rows, h, scores, s, inner, r

# 2 Perform a grid search

# 3 Visualise grid search results
