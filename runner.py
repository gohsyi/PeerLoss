import logger
import multiprocessing as mp

from parser import parse_args
from utils import *
from model import BinaryClassifier
from dataloader import UCIDataLoader, make_noisy_data
from results_plotter import plot

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def train(kargs):
    set_global_seeds(kargs['seed'])
    dataset = UCIDataLoader(kargs['dataset'])
    dataset.equalize_prior()
    X_train, X_test, y_train, y_test = dataset.split_and_normalize()
    y_noisy = make_noisy_data(y_train, kargs['e0'], kargs['e1'])
    classifier = BinaryClassifier(X_train.shape[-1], kargs['lr'], kargs['hidsize'])
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        peer_loss=kargs['peer_loss'],
        batchsize=kargs['batchsize'],
        episodes=kargs['episodes'],
    )

    return test_acc


def get_max_mean(result):
    return max([np.mean(result[-i - 10:-i - 1]) for i in range(0, len(result) - 10)])


def run(arg_dict):
    logger.info(arg_dict)
    pool = mp.Pool(mp.cpu_count())

    # nn with peer loss
    args = []
    arg_dict.update({'peer_loss': True})
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_peer = pool.map(train, args)

    # nn with bce loss
    args = []
    arg_dict.update({'peer_loss': False})
    for seed in range(32):
        arg_dict.update({'seed': seed})
        args.append(arg_dict.copy())
    results_bce = pool.map(train, args)

    # prepare data for traditional methods
    dataset = UCIDataLoader(arg_dict['dataset'])
    dataset.equalize_prior()
    X_train, X_test, y_train, y_test = dataset.split_and_normalize()
    y_noisy = make_noisy_data(y_train, arg_dict['e0'], arg_dict['e1'])

    # support vector machine
    svm_acc = SVC(gamma='auto').fit(X_train, y_noisy).score(X_test, y_test)

    # logistic regression
    lr_acc = LogisticRegression(solver='lbfgs').fit(X_train, y_noisy).score(X_test, y_test)

    # random forest
    rf_acc = RandomForestClassifier(n_estimators=100).fit(X_train, y_noisy).score(X_test, y_test)

    # knn
    knn_acc = KNeighborsClassifier().fit(X_train, y_noisy).score(X_test, y_test)

    plot([results_peer, results_bce], ['use peer loss', 'use bce loss'])

    logger.record_tabular('nn with peer prediction', get_max_mean(results_peer))
    logger.record_tabular('nn with bce', get_max_mean(results_bce))
    logger.record_tabular('svm', svm_acc)
    logger.record_tabular('logistic regression', lr_acc)
    logger.record_tabular('random forest', rf_acc)
    logger.record_tabular('knn', knn_acc)
    logger.dump_tabular()


if __name__ == '__main__':
    run(parse_args().__dict__)
