from openml.datasets import list_datasets, get_datasets
from pandas.api import types


# 1 List dataset info
datalist = list_datasets(number_instances='100..200',
                         number_features='4..100',
                         output_format='dataframe')

# 2 Load datasets
datasets = get_datasets(datalist['did'])

# 3 Filter datasets
datasets_temp = {}
for ds in datasets:

    X, _, nominal, _ = ds.get_data()
    y_name = ds.default_target_attribute
    ds_name = ds.name

    check = (y_name and  # y_name is not None
             ',' not in y_name and  # only 1 target variable y
             not types.is_numeric_dtype(X[y_name]) and  # y is nominal
             sum(nominal) == 1 and  # no other variable is nominal
             ds_name not in datasets_temp)  # remove duplicate datasets

    if check:
        datasets_temp[ds_name] = ds

datasets = datasets_temp
del ds, X, nominal, y_name, ds_name, check, datasets_temp
