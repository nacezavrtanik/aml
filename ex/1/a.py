import pandas as pd
import matplotlib as mpl

mpl.use('TkAgg')  # To avoid TypeError when calling pd.DataFrame.hist in PyCharm


if __name__ == '__main__':

    # 1 Load data
    data = pd.read_csv('podatki.csv')

    # 2 Extract basic information
    rows, columns = data.shape
    data_types = data.dtypes
    stats = data.describe()
    nans = data.isnull().sum()

    # 3 Handle NaN values

    # Drop columns with more than 1/5 NaN values
    threshold = int(rows - rows / 5)
    data.dropna(axis=1, thresh=threshold, inplace=True)
    del threshold

    # Fill remaining Nan values with mean (numeric) or modus (other)
    fillers = dict(data.mean(numeric_only=True))
    fillers['X4'] = data['X4'].mode()[0]
    data.fillna(value=fillers, inplace=True)
    del fillers

    # 4 Visualise data

    # Plot histogram
    # data.hist(legend=True)

    # Create correlation matrix
    corr = data.corr()
