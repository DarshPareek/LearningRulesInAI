import pandas as pd

def readData(path):
    return pd.read_csv(path, index_col=False)

def genData(path):
    data = readData(path)
    #return (data.shape, list(data.columns), data.to_numpy())
    return (data.shape, list(data.columns), data.to_numpy())


