"""Class 1, Exercise C: Linear Regression"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from aml import compare_models_cross_validation


# 1 Calculate regression metrics

# Make predictions
data = pd.read_csv('podatki_regresija.csv')
X, y = data.iloc[:, :-1], data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', rmse)
r2 = r2_score(y_test, y_pred)
print('R2:', r2)

# 2 Cross-validate and compare models
comparison = compare_models_cross_validation(X, y, 'regression',
                                             ['k_neighbors_regressor',
                                              'linear_regression',
                                              'random_forest_regressor',
                                              'SVR'])
