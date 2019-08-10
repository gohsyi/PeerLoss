import multiprocessing as mp
import numpy as np

from model import BinaryClassifier
from utils.parser import parse_args
from utils.misc import set_global_seeds
from utils.dataloader import DataLoader, make_noisy_data
from utils.results_plotter import plot
from utils import logger

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def run_nn(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    if kargs['equalize_prior']:
        dataset.equalize_prior()
    X_train, X_test, y_train, y_test = dataset.train_test_split(kargs['test_size'], kargs['normalize'])
    y_noisy = make_noisy_data(y_train, kargs['e0'], kargs['e1'])
    classifier = BinaryClassifier(
        feature_dim=X_train.shape[-1],
        lr=kargs['lr'],
        hidsize=kargs['hidsize'],
        dropout=kargs['dropout']
    )
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        peer_loss=kargs['peer_loss'],
        batchsize=kargs['batchsize'],
        episodes=kargs['episodes'],
    )
    return test_acc


def run_svm(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    if kargs['equalize_prior']:
        dataset.equalize_prior()
    X_train, X_test, y_train, y_test = dataset.train_test_split()
    y_noisy = make_noisy_data(y_train, kargs['e0'], kargs['e1'])
    model = SVC(gamma='auto')
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def run_lr(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    if kargs['equalize_prior']:
        dataset.equalize_prior()
    X_train, X_test, y_train, y_test = dataset.train_test_split()
    y_noisy = make_noisy_data(y_train, kargs['e0'], kargs['e1'])
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def run_knn(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    if kargs['equalize_prior']:
        dataset.equalize_prior()
    X_train, X_test, y_train, y_test = dataset.train_test_split()
    y_noisy = make_noisy_data(y_train, kargs['e0'], kargs['e1'])
    model = KNeighborsClassifier()
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def run_rf(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    if kargs['equalize_prior']:
        dataset.equalize_prior()
    X_train, X_test, y_train, y_test = dataset.train_test_split()
    y_noisy = make_noisy_data(y_train, kargs['e0'], kargs['e1'])
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def get_max_mean(result):
    return max([np.mean(result[-i-10:-i-1]) for i in range(0, len(result)-10)])


def run(arg_dict):
    logger.info(arg_dict)
    pool = mp.Pool(mp.cpu_count())

    # nn with peer loss
    args = []
    arg_dict.update({'peer_loss': True})
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_peer = pool.map(run_nn, args)

    # nn with bce loss
    args = []
    arg_dict.update({'peer_loss': False})
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_bce = pool.map(run_nn, args)

    # logistic regression
    args = []
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_lr = pool.map(run_lr, args)

    # svm
    args = []
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_svm = pool.map(run_svm, args)

    # knn
    args = []
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_knn = pool.map(run_knn, args)

    # random forest
    args = []
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_rf = pool.map(run_rf, args)

    plot([results_peer, results_bce], ['use peer loss', 'use bce loss'])

    logger.record_tabular('nn with peer prediction', get_max_mean(np.mean(results_peer, 0)))
    logger.record_tabular('nn', get_max_mean(np.mean(results_bce, 0)))
    logger.record_tabular('logistic regression', np.mean(results_lr))
    logger.record_tabular('svm', np.mean(results_svm))
    logger.record_tabular('knn', np.mean(results_knn))
    logger.record_tabular('random forest', np.mean(results_rf))
    logger.dump_tabular()


if __name__ == '__main__':
    run(parse_args().__dict__)
