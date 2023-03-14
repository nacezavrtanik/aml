"""Class 3, Exercise A: Meta-classification, Meta-regression"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

from aml import compare_models_cross_validation, features_by_importance


# 1 Preprocess data

# Read data
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)

X_meta_all = pd.read_csv(parent_dir + '\\2\\c_meta_features.csv', index_col=0).set_index('dataset')
y_meta_all = pd.read_csv(
    parent_dir + '\\2\\b_model_accuracy_comparison_for_datasets.csv', index_col=0).set_index('dataset')
del abs_path, file_dir, parent_dir

# Clean data
X_meta = X_meta_all.dropna(axis=1)
y_meta = y_meta_all['best']  # keep categorical variable only

# Split dataset
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_meta, y_meta, train_size=0.8, random_state=0)

# 2 Cross-validate and compare meta-models
comparison_cls = compare_models_cross_validation(X_meta, y_meta, 'classification',
                                                 ['random_forest_classifier',
                                                  'dummy_classifier'])

# 3 Compare features by importance
important_features = features_by_importance(X_meta, y_meta, random_state=0, plot=False)

# 4 Use a regression meta-model to predict accuracy
comparison_reg = {}
for model in ['k_neighbors_classifier', 'decision_tree_classifier', 'gaussian_nb']:

    comparison_reg[model] = compare_models_cross_validation(X_meta, y_meta_all[model], 'regression',
                                                            ['dummy_regressor',
                                                             'random_forest_regressor'])
