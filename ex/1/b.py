from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from a import col_trans


# 1 Train model on entire dataset
knc = KNeighborsClassifier()
X, y = col_trans[:, 0:-1], col_trans[:, -1]
knc.fit(X, y)

# 2 Evaluate accuracy of model
print('Score (entire dataset):', knc.score(X, y))

# 3 Split dataset into train data and test data, retrain, and reevaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
knc.fit(X_train, y_train)
print('Score (train/test split):', knc.score(X_test, y_test))
