import random
import numpy as np
import pandas as pd
import torch


def set_global_seeds(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)


def load_data(name):
    return pd.read_csv(f'uci/{name}.csv')


def equalize_prior(df, target='target'):
    pos = df.loc[df[target] == 1]
    neg = df.loc[df[target] == 0]
    n = min(pos.shape[0], neg.shape[0])
    pos = pos.sample(n=n)
    neg = neg.sample(n=n)
    df = pd.concat([pos, neg], axis=0)
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
