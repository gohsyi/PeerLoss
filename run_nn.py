import datetime
import numpy as np
import multiprocessing as mp

from utils import logger
from utils.dataloader import DataLoader
from utils.results_plotter import plot, plot__
from utils.misc import set_global_seeds, make_arg_list

from models.nn import MLP, BinaryClassifier, PeerBinaryClassifier, SurrogateBinaryClassifier

BATCH_SIZES = [1, 4, 16, 64]
HIDSIZES = [8, 16, 32, 64]
LEARNING_RATES = [0.0005, 0.001, 0.005, 0.01, 0.05]


def find_best_params(args, pool):
    results = np.empty((len(BATCH_SIZES), len(LEARNING_RATES), len(HIDSIZES)))

    for i, batchsize in enumerate(BATCH_SIZES):
        for j, lr in enumerate(LEARNING_RATES):
            for k, hidsize in enumerate(HIDSIZES):
                args.update({
                    'batchsize': batchsize,
                    'hidsize': [hidsize, hidsize],
                    'lr': lr
                })
                res = [res['val_acc'] for res in pool.map(run_nn_peer, make_arg_list(args, 8))]
                res = np.mean(res, axis=0)
                results[i, j, k] = np.mean(res[-100:])
                if 'verbose' in args.keys() and args['verbose']:
                    print(f'batchsize:{batchsize:2}\tlr:{lr:6}\thidsize:{hidsize:3}\tacc:{results[i, j, k]:.3}')

    best_batchsize, best_lr, best_hidsize = np.unravel_index(results.reshape(-1).argmax(), results.shape)
    best_batchsize = BATCH_SIZES[best_batchsize]
    best_lr = LEARNING_RATES[best_lr]
    best_hidsize = HIDSIZES[best_hidsize]

    return {
        'batchsize': best_batchsize,
        'hidsize': [best_hidsize, best_hidsize],
        'lr': best_lr,
    }


def run_nn(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    classifier = BinaryClassifier(
        model=MLP(feature_dim=X_train.shape[-1], hidsizes=args['hidsize'], dropout=args['dropout']),
        learning_rate=args['lr'],
        loss_func=args['loss'],
    )
    results = classifier.fit(
        X_train, y_train, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
    )
    return results


def run_nn_surr(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    classifier = SurrogateBinaryClassifier(
        model=MLP(feature_dim=X_train.shape[-1], hidsizes=args['hidsize'], dropout=args['dropout']),
        learning_rate=args['lr'],
        loss_func=args['loss'],
        e0=args['e0'],
        e1=args['e1'],
    )
    results = classifier.fit(
        X_train, y_train, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
    )
    return results


def run_nn_peer(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    classifier = PeerBinaryClassifier(
        model=MLP(feature_dim=X_train.shape[-1], hidsizes=args['hidsize'], dropout=args['dropout']),
        learning_rate=args['lr'],
        loss_func=args['loss'],
        alpha=args['alpha'],
    )
    results = classifier.fit(
        X_train, y_train, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
    )
    return results


def get_max_mean(result, interval=100):
    return max([np.mean(result[-i-interval:-i-1]) for i in range(0, len(result)-interval)])


def run(args):
    logger.configure(f'logs/nn/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    logger.info(args)

    pool = mp.Pool(mp.cpu_count())

    nn_arg = args.copy()
    if 'batchsize' not in nn_arg.keys() or 'lr' not in nn_arg.keys() or 'hidsize' not in nn_arg.keys():
        nn_arg.update(find_best_params(nn_arg, pool))
        logger.record_tabular('[NN] best batchsize', nn_arg['batchsize'])
        logger.record_tabular('[NN] best learning rate', nn_arg['lr'])
        logger.record_tabular('[NN] best hidsize', nn_arg['hidsize'])
        logger.dump_tabular()

    results_surr = pool.map(run_nn_surr, make_arg_list(nn_arg))
    results_nn = pool.map(run_nn, make_arg_list(nn_arg))
    results_peer = pool.map(run_nn_peer, make_arg_list(nn_arg))

    test_acc_bce = [res['val_acc'] for res in results_nn]
    test_acc_peer = [res['val_acc'] for res in results_peer]
    test_acc_surr = [res['val_acc'] for res in results_surr]

    plot([test_acc_bce, test_acc_peer, test_acc_surr],
         ['cross entropy loss', 'peer loss', 'surrogate loss'],
         title='Accuracy During Testing')

    train_acc_bce = [res['train_acc'] for res in results_nn]
    train_acc_peer = [res['train_acc'] for res in results_peer]
    train_acc_surr = [res['train_acc'] for res in results_surr]

    plot([train_acc_bce, train_acc_peer, train_acc_surr],
         ['cross entropy loss', 'peer loss', 'surrogate loss'],
         title='Accuracy During Training')

    loss_acc_surr = [res['loss'] for res in results_surr]
    loss_acc_bce = [res['loss'] for res in results_nn]
    loss_acc_peer = [res['loss'] for res in results_peer]

    plot([loss_acc_bce, loss_acc_peer, loss_acc_surr],
         ['cross entropy loss', 'peer loss', 'surrogate loss'],
         title='Loss')

    logger.record_tabular('[NN] with peer loss', np.mean(test_acc_peer, 0)[-100:].mean())
    logger.record_tabular('[NN] with surrogate loss', np.mean(test_acc_surr, 0)[-100:].mean())
    logger.record_tabular(f'[NN] with {args["loss"]} loss', np.mean(test_acc_bce, 0)[-100:].mean())
    logger.dump_tabular()


if __name__ == '__main__':
    # from utils.parser import parse_args
    # run(parse_args().__dict__)
    # from runner import run

    run({
        'dataset': 'german',
        'e0': 0.1, 'e1': 0.3,
        'episodes': 3000,
            'batchsize': 4,
            'lr': 0.001,
            'hidsize': [16, 16],
        'test_size': 0.15,
        'val_size': 0.15,
        'dropout': 0,
        'normalize': True,
        'equalize_prior': False,
        'alpha': 1.,
        'loss': 'bce',
        'activation': 'relu',
        'verbose': True,
    })
