import pandas as pd

# 1 Load data
data = pd.read_csv('podatki.csv')

# 2 Print basic information
print('Rows, Columns:\n', data.shape, '\n')
print('Data Types:\n', data.dtypes, '\n')
print('Statistics:\n', data.describe(), '\n')
print('NaN Values per Column:\n', data.isnull().sum(), '\n')


# 3 Handle missing values
def prune_dataset(df, ratio=0.2):

    r, _ = df.shape
    nans = data.isnull().sum()

    cols = []
    for c, n in zip(df.columns, nans):
        if n / r <= ratio:
            cols.append(c)

    df = df[cols]

    return df
