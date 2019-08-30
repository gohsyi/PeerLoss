import datetime
import numpy as np
import multiprocessing as mp

from utils import logger
from utils.dataloader import DataLoader
from utils.results_plotter import plot, plot__
from utils.misc import set_global_seeds, make_arg_list

from models.nn import MLP, BinaryClassifier, PeerBinaryClassifier, SurrogateBinaryClassifier

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BATCH_SIZES = [1, 4, 16, 32, 64]
HIDSIZES = [[8, 8], [16, 16], [32, 32], [64, 64]]
LEARNING_RATES = [0.0007, 0.001, 0.005, 0.01, 0.05]


def find_best_alpha(args, alphas=ALPHAS):
    pool = mp.Pool(mp.cpu_count())
    results = []
    for alpha in alphas:
        args['alpha'] = alpha
        res = [res['val_acc'] for res in pool.map(run_nn_peer, make_arg_list(args, 8))]
        res = np.mean(res, axis=0)[-1]
        if 'verbose' in args.keys() and args['verbose']:
            logger.record_tabular(f'[PEER] alpha = {alpha}', res)
        results.append(res)
    pool.close()
    pool.join()
    logger.dump_tabular()
    best_alpha = alphas[np.argmax(results)]
    return {'alpha': best_alpha}


def find_best_alpha_val(args, alphas=ALPHAS):
    pool = mp.Pool(mp.cpu_count())
    results = []
    for alpha in alphas:
        args['alpha'] = alpha
        res = [res['val_acc'] for res in pool.map(run_nn_peer_val, make_arg_list(args, 8))]
        res = np.mean(res, axis=0)
        res = np.mean(res[-1])
        if 'verbose' in args.keys() and args['verbose']:
            logger.record_tabular(f'[PEER] alpha = {alpha}', res)
        results.append(res)
    pool.close()
    pool.join()
    logger.dump_tabular()
    best_alpha = alphas[np.argmax(results)]
    return {'alpha': best_alpha}


def find_best_params(args, batchsizes=None, lrs=None, hidsizes=None):
    args = args.copy()
    if 'alpha' not in args.keys():
        args['alpha'] = 1.0

    pool = mp.Pool(mp.cpu_count())
    results = np.empty((len(batchsizes), len(lrs), len(hidsizes)))

    batchsizes = batchsizes or [args['batchsize']]
    lrs = lrs or [args['lr']]
    hidsizes = hidsizes or [args['hidsize']]

    for k, hidsize in enumerate(hidsizes):
        for i, batchsize in enumerate(batchsizes):
            for j, lr in enumerate(lrs):
                args.update({
                    'batchsize': batchsize,
                    'hidsize': hidsize,
                    'lr': lr
                })
                res = [res['val_acc'] for res in pool.map(run_nn_peer, make_arg_list(args, 8))]
                res = np.mean(res, axis=0)
                results[i, j, k] = np.mean(res[-1])
                if 'verbose' in args.keys() and args['verbose']:
                    print(f'acc:{results[i, j, k]:4.3}\t'
                          f'hidsize:{str(hidsize):8}\t'
                          f'batchsize:{batchsize:2}\t'
                          f'lr:{lr:6}\t')
    pool.close()
    pool.join()
    best_batchsize, best_lr, best_hidsize = np.unravel_index(results.reshape(-1).argmax(), results.shape)
    best_acc = results.max()
    best_batchsize = batchsizes[best_batchsize]
    best_lr = lrs[best_lr]
    best_hidsize = hidsizes[best_hidsize]

    return {
        'batchsize': best_batchsize,
        'hidsize': best_hidsize,
        'lr': best_lr,
        'acc': best_acc,
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


def run_nn_peer_val(args):
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
        X_train, y_train, X_val, y_val,
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
        nn_arg.update(find_best_params(nn_arg, BATCH_SIZES, LEARNING_RATES, HIDSIZES))
        logger.record_tabular('[PEER] best batchsize', nn_arg['batchsize'])
        logger.record_tabular('[PEER] best learning rate', nn_arg['lr'])
        logger.record_tabular('[PEER] best hidsize', nn_arg['hidsize'])
        logger.dump_tabular()

    if 'alpha' not in nn_arg.keys():
        nn_arg.update(find_best_alpha_val(nn_arg))
        logger.record_tabular('[PEER] best alpha', nn_arg['alpha'])
    elif type(nn_arg['alpha']) == list or type(nn_arg['alpha']) == np.ndarray:
        nn_arg.update(find_best_alpha_val(nn_arg, nn_arg['alpha']))
        logger.record_tabular('[PEER] best alpha', nn_arg['alpha'])

    results_surr = pool.map(run_nn_surr, make_arg_list(nn_arg))
    results_nn = pool.map(run_nn, make_arg_list(nn_arg))
    results_peer = pool.map(run_nn_peer, make_arg_list(nn_arg))
    pool.close()
    pool.join()

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

    logger.record_tabular('[NN] with peer loss', np.mean(test_acc_peer, 0)[-1])
    logger.record_tabular('[NN] with surrogate loss', np.mean(test_acc_surr, 0)[-1])
    logger.record_tabular(f'[NN] with {args["loss"]} loss', np.mean(test_acc_bce, 0)[-1])
    logger.dump_tabular()


if __name__ == '__main__':
    from utils.parser import parse_args
    run(parse_args().__dict__)
