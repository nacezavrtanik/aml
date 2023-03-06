"""Class 2, Exercise C: Preparing Features"""

from pymfe.mfe import MFE
from sklearn.tree import DecisionTreeClassifier

from a import datasets


# 1 Extract meta-features from data
ds = datasets.get('wine')
X_all, _, _, _ = ds.get_data()
y_name = ds.default_target_attribute
X, y = X_all.copy().drop(y_name, axis=1).to_numpy(), X_all[y_name].to_numpy()  # X, y must be numpy arrays!

mfe = MFE(groups=['general', 'info-theory'])
mfe.fit(X, y)
features_from_data = mfe.extract()

# 2 Extract meta-features from model
dtc = DecisionTreeClassifier()
dtc.fit(X, y)
mfe = MFE(groups=['model-based'])
features_from_model = mfe.extract_from_model(dtc)
