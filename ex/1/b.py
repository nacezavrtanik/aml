"""Class 1, Exercise B: Binary Classification"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
# from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from a import col_trans
# from aml import plot_validation_curve


# 1 Train model on entire dataset
knc = KNeighborsClassifier()
X, y = col_trans[:, 0:-1], col_trans[:, -1]
knc.fit(X, y)

# 2 Evaluate accuracy of model
print('Accuracy (entire dataset, not scaled):', knc.score(X, y))

# 3 Split dataset into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
knc.fit(X_train, y_train)
print('Accuracy (not scaled):', knc.score(X_test, y_test))

# 4 Scale features, analyze hyperparameters

# Standardise features
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
knc.fit(X_train_scaled, y_train)
print('Accuracy:', knc.score(X_test_scaled, y_test))

# Visualise effect of hyperparameter n_neighbors
_param_name = 'n_neighbors'
_param_range = range(1, 31)
train_scores, test_scores = validation_curve(
    knc, X_train_scaled, y_train, param_name=_param_name, param_range=_param_range)
# plot_validation_curve(
#     train_scores, test_scores, _param_range, title='Validation Curve for KNeighborsClassifier', xlabel=_param_name)
del _param_name, _param_range, train_scores, test_scores

# 5 Calculate alternative metrics

# Train model with different hyperparameters
_n_neighbors = 15
knc = KNeighborsClassifier(n_neighbors=_n_neighbors)
knc.fit(X_train_scaled, y_train)
y_pred = knc.predict(X_test_scaled)
y_score = knc.predict_proba(X_test_scaled)[:, 1]
print(f'Accuracy (n_neighbors={_n_neighbors}):', knc.score(X_test_scaled, y_test))
del _n_neighbors

# Confusion matrix
confusion = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion).plot()

# Precision, recall
precision, recall, _ = precision_recall_curve(y_test, y_score)
# PrecisionRecallDisplay(precision, recall).plot()
del precision, recall

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_score)
# RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
del fpr, tpr

# ROC AUC
auc_score = roc_auc_score(y_test, y_score)
print('AUC ROC:', auc_score)
