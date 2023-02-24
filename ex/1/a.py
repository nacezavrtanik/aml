import pandas as pd


if __name__ == '__main__':

    # 1 Load data
    data = pd.read_csv('podatki.csv')

    # 2 Print basic information
    print('Rows, Columns:\n', data.shape, '\n')
    print('Data Types:\n', data.dtypes, '\n')
    print('Statistics:\n', data.describe(), '\n')
    print('NaN Values per Column:\n', data.isnull().sum(), '\n')

    # 3 Handle missing values

    # Drop columns with more than 1/5 NaN values
    rows, _ = data.shape
    threshold = int(rows - rows / 5)
    data.dropna(axis=1, thresh=threshold, inplace=True)

    # Fill remaining Nan values with mean (numeric) or modus (other)
    fillers = dict(data.mean(numeric_only=True))
    fillers['X4'] = data['X4'].mode()[0]
    data.fillna(value=fillers, inplace=True)
