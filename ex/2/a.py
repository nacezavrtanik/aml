from openml.datasets import list_datasets, get_datasets


# 1 Filter datasets
datalist = list_datasets(number_instances='100..200',
                         number_features='4..100',
                         output_format='dataframe')

# 2 Load datasets
