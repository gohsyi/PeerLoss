import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class UCIDataLoader(object):
    def __init__(self, name):
        if name == 'heart':
            self.df = pd.read_csv(f'uci/heart.csv')
            self.df = preprocess_heart(self.df)
        elif name == 'breast':
            self.df = pd.read_csv(f'uci/breast.csv')
            self.df = preprocess_breast(self.df)
        elif name == 'german':
            self.df = pd.read_csv(f'uci/heart.csv')
        elif name == 'banana':
            self.df = pd.read_csv(f'uci/banana.csv')

    def load(self, path):
        df = open(path).readlines()
        df = list(map(lambda line: list(map(int, line.split())), df))
        self.df = pd.DataFrame(df)
        return self

    def equalize_prior(self, target='target'):
        pos = self.df.loc[self.df[target] == 1]
        neg = self.df.loc[self.df[target] == 0]
        n = min(pos.shape[0], neg.shape[0])
        pos = pos.sample(n=n)
        neg = neg.sample(n=n)
        self.df = pd.concat([pos, neg], axis=0)
        return self

    def split_and_normalize(self, test_size=0.25):
        X = self.df.drop(['target'], axis=1).values
        y = self.df.target.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        return X_train, X_test, y_train, y_test


def onehot(df, cols):
    dummies = [pd.get_dummies(df[col]) for col in cols]
    df.drop(cols, axis=1, inplace=True)
    df = pd.concat([df] + dummies, axis=1)
    return df


def preprocess_heart(df):
    df = onehot(df, ['cp', 'slope', 'thal', 'restecg'])
    return df


def preprocess_breast(df):
    df.replace({'M': 1, 'B': 0}, inplace=True)
    df.rename(columns={'diagnosis': 'target'}, inplace=True)
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
    return df


def make_noisy_data(y, e0, e1):
    num_neg = np.count_nonzero(y == 0)
    num_pos = np.count_nonzero(y == 1)
    flip0 = np.random.choice(np.where(y == 0)[0], int(num_neg * e0), replace=False)
    flip1 = np.random.choice(np.where(y == 1)[0], int(num_pos * e1), replace=False)
    flipped_idxes = np.concatenate([flip0, flip1])
    y_noisy = y.copy()
    y_noisy[flipped_idxes] = 1 - y_noisy[flipped_idxes]
    return y_noisy
