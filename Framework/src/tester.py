import pandas as pd




def readData():
    return pd.read_csv("test.csv", index_col=False)

def genData():
    data = readData()
    return (data.shape, list(data.columns), data.to_numpy())

print(type(genData()))
