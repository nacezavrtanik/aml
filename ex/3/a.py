"""Class 3, Exercise A: Meta-classification, Meta-regression"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split


# 1 Preprocess data

# Read data
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)

X_meta = pd.read_csv(parent_dir + '\\2\\c_meta_features.csv', index_col=0)
y_meta = pd.read_csv(parent_dir + '\\2\\b_model_accuracy_comparison_for_datasets.csv', index_col=0)

# Clean data
X_meta.dropna(axis=1, inplace=True)
X_meta.drop(columns=['nr_cat', 'cat_to_num'], inplace=True)  # contained only zeros, as we removed categorical features

# Split dataset
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_meta, y_meta, train_size=0.8, random_state=0)
