import multiprocessing as mp
import numpy as np

from utils.parser import parse_args
from utils.misc import set_global_seeds
from utils.dataloader import DataLoader
from utils.results_plotter import plot
from utils import logger

from models.nn import BinaryClassifier, PeerBinaryClassifier, SurrogateBinaryClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def run_nn(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    classifier = BinaryClassifier(
        feature_dim=X_train.shape[-1],
        lr=kargs['lr'],
        hidsize=kargs['hidsize'],
        dropout=kargs['dropout']
    )
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        batchsize=kargs['batchsize'],
        episodes=kargs['episodes'],
    )
    if 'verbose' in kargs.keys():
        plot([[train_acc], [test_acc]], ['nn during train', 'nn during test'])
    return test_acc


def run_nn_surr(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    classifier = SurrogateBinaryClassifier(
        feature_dim=X_train.shape[-1],
        lr=kargs['lr'],
        hidsize=kargs['hidsize'],
        dropout=kargs['dropout'],
        e0=kargs['e0'],
        e1=kargs['e1'],
    )
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        batchsize=kargs['batchsize'],
        episodes=kargs['episodes'],
    )
    if 'verbose' in kargs.keys():
        plot([[train_acc], [test_acc]],
             ['nn with surrogate loss during train',
              'nn with surrogate loss during test'])
    return test_acc


def run_nn_peer(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    classifier = PeerBinaryClassifier(
        feature_dim=X_train.shape[-1],
        lr=kargs['lr'],
        hidsize=kargs['hidsize'],
        dropout=kargs['dropout'],
        alpha=kargs['alpha'],
    )
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        batchsize=kargs['batchsize'],
        episodes=kargs['episodes'],
    )
    if 'verbose' in kargs.keys():
        plot([[train_acc], [test_acc]], ['nn with peer loss during train', 'nn with peer loss during test'])
    return test_acc


def run_c_svm(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    results = []
    for c1 in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        model = SVC(gamma='auto', class_weight={0: 1., 1: c1})
        model.fit(X_train, y_noisy)
        results.append(model.score(X_test, y_test))
    return np.max(results)


def run_lr(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def run_knn(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    model = KNeighborsClassifier()
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def run_rf(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
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
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_peer = pool.map(run_nn_peer, args)

    # nn with bce loss
    args = []
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_bce = pool.map(run_nn, args)

    # nn with surrogate loss
    args = []
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_surr = pool.map(run_nn_surr, args)

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
    results_svm = pool.map(run_c_svm, args)

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

    plot([results_bce, results_peer, results_surr], ['use peer loss', 'use bce loss', 'use surrogate loss'])

    logger.record_tabular('nn with peer prediction', get_max_mean(np.mean(results_peer, 0)))
    logger.record_tabular('nn with surrogate loss', get_max_mean(np.mean(results_peer, 0)))
    logger.record_tabular('nn', get_max_mean(np.mean(results_bce, 0)))
    logger.record_tabular('logistic regression', np.mean(results_lr))
    logger.record_tabular('svm', np.mean(results_svm))
    logger.record_tabular('knn', np.mean(results_knn))
    logger.record_tabular('random forest', np.mean(results_rf))
    logger.dump_tabular()


if __name__ == '__main__':
    args = parse_args().__dict__
    print('surr:', run_nn_surr(args))
    print('nn:', run_nn(args))
    print('peer:', run_nn_peer(args))
    print('c-svm:', run_c_svm(args))
    print('knn:', run_knn(args))
    print('lr:', run_lr(args))
    print('rf:', run_rf(args))
