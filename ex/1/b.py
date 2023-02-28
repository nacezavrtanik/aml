from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler

from a import col_trans
from helpers import plot_validation_curve


# 1 Train model on entire dataset
knc = KNeighborsClassifier()
X, y = col_trans[:, 0:-1], col_trans[:, -1]
knc.fit(X, y)

# 2 Evaluate accuracy of model
print('Score (entire dataset):', knc.score(X, y))

# 3 Split dataset into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
knc.fit(X_train, y_train)
print('Score (train/test split):', knc.score(X_test, y_test))

# 4 Feature scaling and hyperparameters

# Standardise features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Visualise effect of hyperparameter n_neighbors
_param_name = 'n_neighbors'
_param_range = range(1, 31)
train_scores, test_scores = validation_curve(
    knc, X_train_scaled, y_train, param_name=_param_name, param_range=_param_range)
plot_validation_curve(
    train_scores, test_scores, _param_range, title='Validation Curve for KNeighborsClassifier', xlabel=_param_name)
del _param_name, _param_range
