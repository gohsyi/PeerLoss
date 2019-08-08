import pandas as pd


def preprocess_heart(df):
    cp = pd.get_dummies(df['cp'], prefix='cp', drop_first=True)
    slope = pd.get_dummies(df['slope'], prefix='slope')
    thal = pd.get_dummies(df['thal'], prefix='thal')
    restecg = pd.get_dummies(df['restecg'], prefix='restecg')

    df.drop(['cp', 'slope', 'thal', 'restecg'], axis=1, inplace=True)
    df = pd.concat([df, cp, slope, thal, restecg], axis=1)
    return df
