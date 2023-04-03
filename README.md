<h1 align="center">
  <br>
  <a href="https://www.fmf.uni-lj.si/en/"><img src="http://phd.fmf.uni-lj.si/img/logo.gif" width="300"></a>
  <br>
  Advanced Machine Learning
  <br>
</h1>

<p align="center">
  <a href="https://www.fmf.uni-lj.si/sl/imenik/238/todorovski-ljupco/">
    <img src="https://img.shields.io/badge/Professor-Ljup%C4%8Do%20Todorovski-red">
  </a>
  <a href="https://www.fmf.uni-lj.si/sl/imenik/1402/brence-jure/">
    <img src="https://img.shields.io/badge/Assistant-Jure%20Brence-red">
  </a>
  <a href="https://www.fmf.uni-lj.si/sl/imenik/273/osojnik-aljaz/">
    <img src="https://img.shields.io/badge/Assistant-Alja%C5%BE%20Osojnik-red">
  </a>
  <a href="https://www.fmf.uni-lj.si/sl/imenik/177/petkovic-matej/">
    <img src="https://img.shields.io/badge/Assistant-Matej%20Petkovi%C4%87-red">
  </a>
  <a href="https://github.com/nacezavrtanik">
    <img src="https://img.shields.io/badge/RepositoryOwner-Nace%20Zavrtanik-lightgrey">
  </a>
</p>


This is my personal repository for the course Advanced Machine Learning (Napredno strojno učenje) as taken at the
Faculty of Mathematics and Physics, University of Ljubljana, in the 2<sup>nd</sup> semester of 2022/2023. This
repository contains, of course, the code I wrote during the course of this course, such as code from exercise classes
and homework assignments.


<p align="center">
    <img src="https://img.shields.io/badge/python-3.10-9cf">
    <img src="https://img.shields.io/badge/venv-requirements.txt-9cf">
</p>


## Table of Contents

- [Exercise Classes](#exercise-classes)
  - [Class 1: Fundamentals of Machine Learning in Python](#class-1-fundamentals-of-machine-learning-in-python)
  - [Class 2: Meta-learning](#class-2-meta-learning)
  - [Class 3: Meta-learning, and Hyperparameter Optimisation](#class-3-meta-learning-and-hyperparameter-optimisation)
  
- [Homework Assignments](#homework-assignments)
  - [Assignment 1: Method Selection, Hyperparameter Optimisation, Meta-learning](#assignment-1-method-selection-hyperparameter-optimisation-meta-learning)


## Exercise Classes

Below, a broad overview of exercise classes is given. The exact instructions are not part of this repository.
Corresponding code can be found in appropriate subdirectories of `ex/`.


### Class 1: Fundamentals of Machine Learning in Python

- Exercise A: ***Data Processing***
  1. Load data
  2. Extract basic statistics
  3. Handle `NaN` values
  4. Visualise data
  5. Encode categorical features
     - `sklearn.preprocessing.OneHotEncoder`
     - `sklearn.compose.make_column_transformer`
- Exercise B: ***Binary Classification***
  1. Train model on entire dataset
     - `sklearn.neighbors.KNeighborsClassifier`
  2. Evaluate accuracy of model
  3. Split dataset into train data and test data
     - `sklearn.model_selection.train_test_split`
  4. Scale features, analyze hyperparameters
     - `sklearn.preprocessing.StandardScaler`
     - `sklearn.model_selection.validation_curve`
  5. Calculate alternative metrics
     - `sklearn.metrics.confusion_matrix`
     - `sklearn.metrics.precision_recall_curve`
     - `sklearn.metrics.roc_curve`
     - `sklearn.metrics.roc_auc_score`
- Exercise C: ***Linear Regression***
  - `sklearn.linear_model.LinearRegression`
  1. Calculate regression metrics
     - `sklearn.metrics.mean_squared_error`
     - `sklearn.metrics.r2_score`
  2. Cross-validate and compare models
     - `sklearn.model_selection.cross_validate`
     - `sklearn.svm.SVR`
     - `sklearn.ensemble.RandomForestRegressor`
     - `sklearn.neighbors.KNeighborsRegressor`

[(Back to top)](#table-of-contents)

### Class 2: Meta-learning

- Exercise A: ***Obtaining Data from [OpenML](https://www.openml.org/)***
  1. List dataset info
     - `openml.datasets.list_datasets`
  2. Load datasets
     - `openml.datasets.get_datasets`
  3. Filter datasets
     - `pandas.api.types.is_numeric`
- Exercise B: ***Preparing Target Variables***
  1. Compare model accuracies for datasets
     - `sklearn.tree.DecisionTreeClassifier`
     - `sklearn.naive_bayes.GaussianNB`
- Exercise C: ***Preparing Meta-features***
  1. Extract meta-features from dataset
     - `pymfe.mfe.MFE.fit`
     - `pymfe.mfe.MFE.extract`
  2. Extract meta-features from fitted model
     - `pymfe.mfe.MFE.extract_from_model`
  3. Extract meta-features for all datasets

[(Back to top)](#table-of-contents)

### Class 3: Meta-learning, and Hyperparameter Optimisation

- Exercise A: ***Meta-classification, Meta-regression***
  1. Preprocess data
  2. Cross-validate and compare meta-models
     - `sklearn.ensemble.RandomForestClassifier`
     - `sklearn.dummy.DummyClassifier`
  3. Compare features by importance
     - `sklearn.ensemble.RandomForestClassifier.feature_importances_`
     - `numpy.argsort`
  4. Use a regression meta-model to predict accuracy
- Exercise B: ***Hyperparameter Optimisation***
  - `sklearn.*.model.get_params`
  1. Vary hyperparameters in a decision tree model
  2. Perform a grid search
     - `sklearn.model_selection.GridSearchCV`
     - `sklearn.model_selection.GridSearchCV.best_params_`
     - `sklearn.model_selection.GridSearchCV.best_estimator_`
  3. Visualise grid search results
     - `grid_search.param_grid`

[(Back to top)](#table-of-contents)


## Homework Assignments

### Assignment 1: Method Selection, Hyperparameter Optimisation, Meta-learning

- Problem 1: ***Method Selection and Hyperparameter Optimisation***
    1. Manual approach
    2. Automated approach
- Problem 2: ***Meta-learning***
    1. Method selection with meta-learning

[(Back to top)](#table-of-contents)
