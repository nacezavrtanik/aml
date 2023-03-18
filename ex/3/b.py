"""Class 3, Exercise B: Hyperparameter Optimisation"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from aml import compare_models_cross_validation, fancy_print, plot_grid_search_results


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

comparison = pd.DataFrame(rows, columns=['max_depth', 'min_samples_split', 'r2']).sort_values(by='r2', ascending=False)
del hyperparams, rows, h, scores, s, inner, r

# 2 Perform a grid search
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)

hyperparams = {'max_depth': [2 + 3*i for i in range(17)],
               'min_samples_split': [5 + 20*j for j in range(20)]}

gs = GridSearchCV(dtr, param_grid=hyperparams, scoring='r2')
gs.fit(X_train, y_train)

best_hyperparams = gs.best_params_
dtr_best = gs.best_estimator_

y_pred, y_pred_best = dtr.predict(X_test), dtr_best.predict(X_test)
r2, r2_best = r2_score(y_test, y_pred), r2_score(y_test, y_pred_best)

fancy_print('r2', r2)
fancy_print('r2_best', r2_best)

# 3 Visualise grid search results
plot_grid_search_results(gs, 'max_depth', 'min_samples_split')
