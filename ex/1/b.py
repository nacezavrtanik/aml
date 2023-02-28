from sklearn.neighbors import KNeighborsClassifier
from a import col_trans


# 1 Train model on entire dataset
knc = KNeighborsClassifier()
X, y = col_trans[:, 0:-1], col_trans[:, -1]
knc.fit(X, y)

# 2 Evaluate accuracy of model
print('Score:', knc.score(X, y))
