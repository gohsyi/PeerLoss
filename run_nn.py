import datetime
import numpy as np
import multiprocessing as mp

from utils import logger
from utils.dataloader import DataLoader
from utils.results_plotter import plot
from utils.misc import set_global_seeds, make_arg_list

from models.nn import MLP, BinaryClassifier, PeerBinaryClassifier, SurrogateBinaryClassifier

ALPHAS = [-5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]


def find_best_alpha(kargs):
    if len(kargs['alpha']) == 1:
        return {
            'alpha': kargs['alpha'][0]
        }
    pool = mp.Pool(mp.cpu_count())
    results = []
    args = kargs.copy()
    for alpha in kargs['alpha']:
        args['alpha'] = alpha
        res = [res['val_acc'] for res in pool.map(run_nn_peer, make_arg_list(args))]
        res = np.mean(res, axis=0)[-1]
        if 'verbose' in args.keys() and args['verbose']:
            logger.record_tabular(f'[PEER] alpha = {alpha}', res)
        results.append(res)
    pool.close()
    pool.join()
    logger.dump_tabular()
    best_alpha = kargs['alpha'][np.argmax(results)]
    return {
        'alpha': best_alpha
    }


def find_best_alpha_val(kargs):
    if len(kargs['alpha']) == 1:
        return {
            'alpha': kargs['alpha'][0]
        }
    args = kargs.copy()
    pool = mp.Pool(mp.cpu_count())
    results = []
    for alpha in kargs['alpha']:
        args['alpha'] = alpha
        res = [res['val_acc'] for res in pool.map(run_nn_peer_val, make_arg_list(args))]
        res = np.mean(res, axis=0)[-1]
        if 'verbose' in args.keys() and args['verbose']:
            logger.record_tabular(f'[PEER] alpha = {alpha}', res)
        results.append(res)
    pool.close()
    pool.join()
    logger.dump_tabular()
    best_alpha = kargs['alpha'][np.argmax(results)]
    return {
        'alpha': best_alpha
    }


def find_best_params(kargs):
    args = kargs.copy()
    args['alpha'] = 1.0
    pool = mp.Pool(mp.cpu_count())
    results = np.empty((len(kargs['batchsize']), len(kargs['lr']), len(kargs['hidsize'])))

    if len(kargs['batchsize']) == 1 and len(kargs['lr']) == 1 and len(kargs['hidsize']) == 1:
        return {
            'batchsize': kargs['batchsize'][0],
            'hidsize': kargs['hidsize'][0],
            'lr': kargs['lr'][0],
        }

    for k, hidsize in enumerate(kargs['hidsize']):
        for i, batchsize in enumerate(kargs['batchsize']):
            for j, lr in enumerate(kargs['lr']):
                args.update({
                    'batchsize': batchsize,
                    'hidsize': hidsize,
                    'lr': lr,
                })
                res = [res['val_acc'] for res in pool.map(run_nn_peer, make_arg_list(args))]
                results[i, j, k] = np.mean(res, axis=0)[-1]
                if 'verbose' in args.keys() and args['verbose']:
                    logger.info(
                        f'acc:{results[i, j, k]:4.3}\t'
                        f'hidsize:{str(hidsize):8}\t'
                        f'batchsize:{batchsize:2}\t'
                        f'lr:{lr:6}\t'
                    )
    pool.close()
    pool.join()
    best_batchsize, best_lr, best_hidsize = np.unravel_index(results.reshape(-1).argmax(), results.shape)
    best_acc = results.max()
    best_batchsize = kargs['batchsize'][best_batchsize]
    best_lr = kargs['lr'][best_lr]
    best_hidsize = kargs['hidsize'][best_hidsize]

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
    mlp = MLP(
        feature_dim=X_train.shape[-1],
        hidsizes=args['hidsize'],
        dropout=args['dropout'],
    )
    classifier = BinaryClassifier(
        model=mlp,
        learning_rate=args['lr'],
        loss_func=args['loss'],
    )
    results = classifier.fit(
        X_train, y_train, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
        logger=logger if args['seeds'] == 1 else None,
    )
    return results


def run_nn_surr(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    mlp = MLP(
        feature_dim=X_train.shape[-1],
        hidsizes=args['hidsize'],
        dropout=args['dropout']
    )
    classifier = SurrogateBinaryClassifier(
        model=mlp,
        learning_rate=args['lr'],
        loss_func=args['loss'],
        e0=args['e0'],
        e1=args['e1'],
    )
    results = classifier.fit(
        X_train, y_train, X_test, y_test,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
        logger=logger if args['seeds'] == 1 else None
    )
    return results


def run_nn_peer(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    mlp = MLP(
        feature_dim=X_train.shape[-1],
        hidsizes=args['hidsize'],
        dropout=args['dropout']
    )
    classifier = PeerBinaryClassifier(
        model=mlp,
        learning_rate=args['lr'],
        loss_func=args['loss'],
        alpha=args['alpha'],
    )
    results = classifier.fit(
        X_train, y_train, X_test, y_test,
        batchsize=args['batchsize'],
        batchsize_=args['batchsize_peer'],
        episodes=args['episodes'],
        logger=logger if args['seeds'] == 1 else None
    )
    return results


def run_nn_peer_val(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    mlp = MLP(
        feature_dim=X_train.shape[-1],
        hidsizes=args['hidsize'],
        dropout=args['dropout']
    )
    classifier = PeerBinaryClassifier(
        model=mlp,
        learning_rate=args['lr'],
        loss_func=args['loss'],
        alpha=args['alpha'],
    )
    results = classifier.fit(
        X_train, y_train, X_val, y_val,
        batchsize=args['batchsize'],
        episodes=args['episodes'],
        logger=logger if args['seeds'] == 1 else None
    )
    return results


def get_max_mean(result, interval=100):
    return max([np.mean(result[-i-interval:-i-1]) for i in range(0, len(result)-interval)])


def run(args):
    prefix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    logger.configure(f'logs/{args["dataset"]}/nn/{prefix}')
    logger.info(args)

    pool = mp.Pool(mp.cpu_count())

    nn_arg = args.copy()
    nn_arg.update(find_best_params(nn_arg))
    nn_arg.update(find_best_alpha_val(nn_arg))
    logger.record_tabular('[PEER] batchsize', nn_arg['batchsize'])
    logger.record_tabular('[PEER] learning rate', nn_arg['lr'])
    logger.record_tabular('[PEER] hidsize', nn_arg['hidsize'])
    logger.record_tabular('[PEER] alpha', nn_arg['alpha'])
    logger.dump_tabular()

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
         title='Accuracy During Testing',
         path=f'logs/{args["dataset"]}/nn/{prefix}')

    train_acc_bce = [res['train_acc'] for res in results_nn]
    train_acc_peer = [res['train_acc'] for res in results_peer]
    train_acc_surr = [res['train_acc'] for res in results_surr]

    plot([train_acc_bce, train_acc_peer, train_acc_surr],
         ['cross entropy loss', 'peer loss', 'surrogate loss'],
         title='Accuracy During Training',
         path=f'logs/{args["dataset"]}/nn/{prefix}')

    loss_acc_surr = [res['loss'] for res in results_surr]
    loss_acc_bce = [res['loss'] for res in results_nn]
    loss_acc_peer = [res['loss'] for res in results_peer]

    plot([loss_acc_bce, loss_acc_peer, loss_acc_surr],
         ['cross entropy loss', 'peer loss', 'surrogate loss'],
         title='Loss',
         path=f'logs/{args["dataset"]}/nn/{prefix}')

    logger.record_tabular('[NN] with peer loss', np.mean(test_acc_peer, 0)[-1])
    logger.record_tabular('[NN] with surrogate loss', np.mean(test_acc_surr, 0)[-1])
    logger.record_tabular(f'[NN] with {args["loss"]} loss', np.mean(test_acc_bce, 0)[-1])
    logger.dump_tabular()


if __name__ == '__main__':
    from utils.parser import parse_args
    run(parse_args().__dict__)
