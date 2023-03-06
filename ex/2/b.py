"""Class 2, Exercise B: Preparing Target Variables"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from a import datasets


# 1 Compare model accuracies for datasets
models = {'k_neighbors_classifier': KNeighborsClassifier(),
          'decision_tree_classifier': DecisionTreeClassifier(),
          'gaussian_nb': GaussianNB()}

scores = {}
for ds_name, ds in datasets.items():

    # Preprocessing
    X_all, _, _, _ = ds.get_data()
    y_name = ds.default_target_attribute
    X, y = X_all.copy().drop(y_name, axis=1), X_all[y_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Create rows for output dataframe
    row = {}
    best = -1
    for md_name, md in models.items():

        # Measure accuracy
        md.fit(X_train, y_train)
        accuracy = md.score(X_test, y_test)
        row[md_name] = accuracy
        if accuracy > best:
            best = accuracy
            row['best'] = md_name

        scores[ds_name] = row
del ds_name, ds, X_all, y_name, X, y, X_train, X_test, y_train, y_test, row, best, md_name, md, accuracy

# Crate dataframe, export to .csv
results = pd.DataFrame.from_dict(scores, orient='index').reset_index(names='dataset')
results = results[['dataset', 'k_neighbors_classifier', 'decision_tree_classifier', 'gaussian_nb', 'best']]
results.to_csv('b_model_accuracy_comparison_for_datasets.csv')
