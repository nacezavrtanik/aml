"""Class 2, Exercise C: Preparing Features"""

import pandas as pd
from pymfe.mfe import MFE
from sklearn.tree import DecisionTreeClassifier

from a import datasets


# 1 Extract meta-features from dataset
ds = datasets.get('wine')
X_all, _, _, _ = ds.get_data()
y_name = ds.default_target_attribute
X, y = X_all.copy().drop(y_name, axis=1).to_numpy(), X_all[y_name].to_numpy()  # X, y must be numpy arrays

mfe = MFE(groups=['general', 'info-theory'])
mfe.fit(X, y)
features_from_data = mfe.extract()

# 2 Extract meta-features from fitted model
dtc = DecisionTreeClassifier()
dtc.fit(X, y)
mfe = MFE(groups=['model-based'])
features_from_model = mfe.extract_from_model(dtc)  # fits MFE instance with data fitted to model, then extracts
del dtc

# 3 Extract meta-features for all datasets
features = {}
for ds_name, ds in datasets.items():

    # Preprocess
    X_all, _, _, _ = ds.get_data()
    y_name = ds.default_target_attribute
    X, y = X_all.copy().drop(y_name, axis=1).to_numpy(), X_all[y_name].to_numpy()

    # Extract meta-features
    mfe = MFE(groups=['general', 'info-theory', 'model-based'])
    mfe.fit(X, y)
    ft_name, ft_value = mfe.extract()
    ds_features = {fn: fv for fn, fv in zip(ft_name, ft_value)}
    features[ds_name] = ds_features
del ds_name, ds, X_all, y_name, X, y, mfe, ft_name, ft_value, ds_features

# Create dataframe, export to .csv
features = pd.DataFrame.from_dict(features, orient='index').reset_index(names='dataset')
features.to_csv('c_meta_features.csv')
