import datetime
import multiprocessing as mp
import numpy as np

from utils.parser import parse_args
from utils.misc import set_global_seeds
from utils.dataloader import DataLoader
from utils.results_plotter import plot
from utils import logger

from models.nn import BinaryClassifier, PeerBinaryClassifier, SurrogateBinaryClassifier
from models.perceptron import Perceptron

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

SEEDS = 16
BATCH_SIZES = [4, 8, 16, 32, 64, 128]
HIDSIZES = [8, 16, 32, 64]
LEARNING_RATES = [0.0005, 0.001, 0.005, 0.01]
MARGINS = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
CLASS_WEIGHTS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def make_arg_list(arg):
    arg = arg.copy()
    args = []
    for seed in range(SEEDS):
        arg['seed'] = seed
        args.append(arg.copy())
    return args


def find_best_params(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    best_acc, best_params = 0, None
    for i, batchsize in enumerate(BATCH_SIZES):
        for j, lr in enumerate(LEARNING_RATES):
            for k, hidsize in enumerate(HIDSIZES):
                classifier = BinaryClassifier(
                    feature_dim=X_train.shape[-1],
                    learning_rate=lr,
                    hidsize=hidsize,
                    dropout=kargs['dropout']
                )
                train_acc, test_acc = classifier.fit(
                    X_train, y_noisy, X_test, y_test,
                    batchsize=batchsize,
                    episodes=kargs['episodes'],
                )
                acc = np.max(test_acc)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (i, j, k)
    return best_params


def run_nn(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    classifier = BinaryClassifier(
        feature_dim=X_train.shape[-1],
        learning_rate=kargs['lr'],
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
        learning_rate=kargs['lr'],
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
        learning_rate=kargs['lr'],
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


def find_best_c1(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    results = []
    for c1 in CLASS_WEIGHTS:
        model = SVC(gamma='auto', class_weight={0: 1., 1: c1})
        model.fit(X_train, y_noisy)
        results.append(model.score(X_test, y_test))
    return np.argmax(results)


def run_c_svm(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    model = SVC(gamma='auto', class_weight={0: 1., 1: kargs['C1']})
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def find_best_margin(kargs, verbose=False):
    """ return `best_margin / 0.1` """
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)

    results = []
    for margin in MARGINS:
        model = Perceptron(feature_dim=X_train.shape[-1], margin=margin)
        model.fit(X_train, y_noisy)
        results.append(model.score(X_test, y_test))
    if verbose:
        for margin, res in zip(MARGINS, results):
            print(f'margin:{margin}\ttest results:{res}')
    return np.argmax(results)


def run_pam(kargs):
    set_global_seeds(kargs['seed'])
    dataset = DataLoader(kargs['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(kargs)
    model = Perceptron(feature_dim=X_train.shape[-1], margin=kargs['margin'])
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


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
    logger.configure(f'logs/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    logger.info(arg_dict)

    pool = mp.Pool(mp.cpu_count())

    # neural network
    nn_arg = arg_dict.copy()
    best_params = pool.map(find_best_params, make_arg_list(nn_arg))
    best_batchsize, best_lr, best_hidsize = zip(*best_params)
    best_batchsize = BATCH_SIZES[np.bincount(best_batchsize).argmax()]
    best_lr = LEARNING_RATES[np.bincount(best_lr).argmax()]
    best_hidsize = LEARNING_RATES[np.bincount(best_hidsize).argmax()]

    nn_arg['batchsize'] = best_batchsize
    nn_arg['lr'] = best_lr
    nn_arg['hidsize'] = best_hidsize

    logger.record_tabular('[NN] best batchsize', best_batchsize)
    logger.record_tabular('[NN] best learning rate', best_lr)
    logger.record_tabular('[NN] best hidsize', best_hidsize)
    logger.dump_tabular()

    results_bce = pool.map(run_nn, make_arg_list(nn_arg))
    results_peer = pool.map(run_nn_peer, make_arg_list(nn_arg))
    results_surr = pool.map(run_nn_surr, make_arg_list(nn_arg))

    # perceptron algorithm with margin (PAM)
    best_margin = pool.map(find_best_margin, make_arg_list(arg_dict))
    best_margin = MARGINS[np.bincount(best_margin).argmax()]
    logger.record_tabular('[PAM] best margin', best_margin)
    logger.dump_tabular()
    pam_arg = arg_dict.copy()
    pam_arg['margin'] = best_margin
    results_pam = pool.map(run_pam, make_arg_list(pam_arg))

    # c-svm (biased svm)
    best_c1 = pool.map(find_best_c1, make_arg_list(arg_dict))
    best_c1 = CLASS_WEIGHTS[np.bincount(best_c1).argmax()]
    logger.record_tabular('[C-SVM] best C1', best_c1)
    logger.dump_tabular()
    svm_arg = arg_dict.copy()
    svm_arg['C1'] = best_c1
    results_svm = pool.map(run_c_svm, make_arg_list(svm_arg))

    """
    # logistic regression
    results_lr = pool.map(run_lr, make_arg_list(arg_dict))
    """

    """
    # knn
    results_knn = pool.map(run_knn, make_arg_list(arg_dict))
    """

    """
    # random forest
    results_rf = pool.map(run_rf, make_arg_list(arg_dict))
    """

    plot([results_bce, results_peer, results_surr], ['use peer loss', 'use bce loss', 'use surrogate loss'])

    logger.record_tabular('nn with peer prediction', get_max_mean(np.mean(results_peer, 0)))
    logger.record_tabular('nn with surrogate loss', get_max_mean(np.mean(results_surr, 0)))
    logger.record_tabular('nn', get_max_mean(np.mean(results_bce, 0)))
    logger.record_tabular('svm', np.mean(results_svm))
    logger.record_tabular('pam', np.mean(results_pam))
    # logger.record_tabular('logistic regression', np.mean(results_lr))
    # logger.record_tabular('knn', np.mean(results_knn))
    # logger.record_tabular('random forest', np.mean(results_rf))
    logger.dump_tabular()


if __name__ == '__main__':
    args = parse_args().__dict__
    run(args)
    # print('surr:', max(run_nn_surr(args)))
    # print('nn:', max(run_nn(args)))
    # print('peer:', max(run_nn_peer(args)))
    # print('c-svm:', run_c_svm(args))
    # print('pam:', run_pam(args))
    # print('knn:', run_knn(args))
    # print('lr:', run_lr(args))
    # print('rf:', run_rf(args))
