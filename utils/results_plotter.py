import os
import os.path as osp
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=str, default='test')
    args = parser.parse_args()
    return args


def plot_(folder):
    results = {'training acc': [], 'testing acc': []}
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            csv = osp.join(root, dir, 'progress.csv')
            df = pd.read_csv(csv)
            for key in results.keys():
                results[key].append(df[key].values)

    for i, key in enumerate(results.keys()):
        mean = np.mean(results[key], axis=0)
        std = np.std(results[key], axis=0)
        results[key] = {'mean': mean, 'std': std}
        _mean = gaussian_filter1d(mean, sigma=3)
        _min = gaussian_filter1d(mean - std, sigma=3)
        _max = gaussian_filter1d(mean + std, sigma=3)
        plt.plot(_mean, label=key, color=f'C{i}')
        plt.fill_between(range(mean.size), _min, _max, color=f'C{i}', alpha=0.3)

    plt.xlabel('episodes')
    plt.legend()
    plt.grid()
    plt.show()

    mean = results['testing acc']['mean']
    max_mean = max([mean[-i-10:-i-1].mean() for i in range(0, len(mean)-10)])
    print(f'The maximal mean test accuracy of 10 episodes is {max_mean}')


def plot(results, labels, title=None):
    for i, (result, label) in enumerate(zip(results, labels)):
        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        _mean = gaussian_filter1d(mean, sigma=3)
        _min = gaussian_filter1d(mean - std, sigma=3)
        _max = gaussian_filter1d(mean + std, sigma=3)
        plt.plot(_mean, label=label, color=f'C{i}')
        plt.fill_between(range(mean.size), _min, _max, color=f'C{i}', alpha=0.3)

    if title:
        plt.title(title)
    plt.xlabel('episodes')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    args = parse_args()
    folder = f'logs/{args.log}'
    plot_(folder)


if __name__ == '__main__':
    main()
