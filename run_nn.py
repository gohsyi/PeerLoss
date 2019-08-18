import datetime
import numpy as np
import multiprocessing as mp

from utils import logger
from utils.dataloader import DataLoader
from utils.results_plotter import plot
from utils.misc import set_global_seeds, make_arg_list

from models.nn import BinaryClassifier, PeerBinaryClassifier, SurrogateBinaryClassifier


BATCH_SIZES = [4, 8, 16, 32, 64, 128]
HIDSIZES = [8, 16, 32, 64]
LEARNING_RATES = [0.0005, 0.001, 0.005, 0.01]


def find_best_params(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)
    best_acc, best_params = 0, None
    for i, batchsize in enumerate(BATCH_SIZES):
        for j, lr in enumerate(LEARNING_RATES):
            for k, hidsize in enumerate(HIDSIZES):
                classifier = BinaryClassifier(
                    feature_dim=X_train.shape[-1],
                    learning_rate=lr,
                    hidsize=hidsize,
                    dropout=args['dropout']
                )
                train_acc, test_acc = classifier.fit(
                    X_train, y_noisy, X_test, y_test,
                    batchsize=batchsize,
                    episodes=args['episodes'],
                )
                acc = np.max(test_acc)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (i, j, k)
    return best_params


def run_nn(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)
    classifier = BinaryClassifier(
        feature_dim=X_train.shape[-1],
        learning_rate=args['lr'],
        hidsize=args['hidsize'],
        dropout=args['dropout']
    )
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
    )
    if 'verbose' in args.keys():
        plot([[train_acc], [test_acc]], ['nn during train', 'nn during test'])
    return test_acc


def run_nn_surr(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)
    classifier = SurrogateBinaryClassifier(
        feature_dim=X_train.shape[-1],
        learning_rate=args['lr'],
        hidsize=args['hidsize'],
        dropout=args['dropout'],
        e0=args['e0'],
        e1=args['e1'],
    )
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
    )
    if 'verbose' in args.keys():
        plot([[train_acc], [test_acc]],
             ['nn with surrogate loss during train',
              'nn with surrogate loss during test'])
    return test_acc


def run_nn_peer(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)
    classifier = PeerBinaryClassifier(
        feature_dim=X_train.shape[-1],
        learning_rate=args['lr'],
        hidsize=args['hidsize'],
        dropout=args['dropout'],
        alpha=args['alpha'],
    )
    train_acc, test_acc = classifier.fit(
        X_train, y_noisy, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
    )
    if 'verbose' in args.keys():
        plot([[train_acc], [test_acc]], ['nn with peer loss during train', 'nn with peer loss during test'])
    return test_acc


def get_max_mean(result):
    return max([np.mean(result[-i-10:-i-1]) for i in range(0, len(result)-10)])


def run(args):
    logger.configure(f'logs/nn/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    logger.info(args)

    pool = mp.Pool(mp.cpu_count())

    nn_arg = args.copy()
    if 'batchsize' not in nn_arg.keys() or 'lr' not in nn_arg.keys() or 'hidsize' not in nn_arg.keys():
        best_params = pool.map(find_best_params, make_arg_list(nn_arg))
        best_batchsize, best_lr, best_hidsize = zip(*best_params)
        best_batchsize = BATCH_SIZES[np.bincount(best_batchsize).argmax()]
        best_lr = LEARNING_RATES[np.bincount(best_lr).argmax()]
        best_hidsize = HIDSIZES[np.bincount(best_hidsize).argmax()]

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

    plot([results_bce, results_peer, results_surr],
         ['use cross entropy loss', 'use peer loss', 'use surrogate loss'])

    logger.record_tabular('[NN] with peer loss', get_max_mean(np.mean(results_peer, 0)))
    logger.record_tabular('[NN] with surrogate loss', get_max_mean(np.mean(results_surr, 0)))
    logger.record_tabular('[NN] with cross entropy loss', get_max_mean(np.mean(results_bce, 0)))
    logger.dump_tabular()


if __name__ == '__main__':
    from utils.parser import parse_args
    run(parse_args().__dict__)
