"""Assignment 1, Problem 2: Meta-learning"""

from openml.datasets import list_datasets

from p1 import calculate_roc_auc_for_random_forest


# 1 Method selection with meta-learning

N_INSTANCES = 1203
N_FEATURES = 32
WEIGHTED_RATIO = 3 * N_INSTANCES / N_FEATURES

datalist = list_datasets(number_instances='600..1800',
                         number_features='20..40',
                         output_format='dataframe')

datalist = datalist[datalist['NumberOfSymbolicFeatures'] == 1].drop_duplicates('name')

delta_non_empty_instances = abs(
    abs(datalist['NumberOfInstances'] - datalist['NumberOfInstancesWithMissingValues']) - N_INSTANCES)
delta_number_of_features = abs(datalist['NumberOfFeatures'] - N_FEATURES)

datalist['metric'] = delta_non_empty_instances + delta_number_of_features * WEIGHTED_RATIO
datalist.sort_values(by='metric', inplace=True)

del N_INSTANCES, N_FEATURES, WEIGHTED_RATIO, delta_non_empty_instances, delta_number_of_features

# BEST MODELS
# 43895 ibm-employee-performance: no tasks
# 1453 PieChart3: weka.RandomForest -- Java random forest implementation, no hyperparams
# 1444 PizzaCutter3: weka.RandomForest -- Java random forest implementation, no hyperparams

roc_score_openml = calculate_roc_auc_for_random_forest(
    {}, print_title='NO HYPERPARAMETERS -- None specified for OpenML run')
