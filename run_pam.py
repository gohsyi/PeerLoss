import datetime
import numpy as np
import multiprocessing as mp

from models.perceptron import Perceptron

from utils import logger
from utils.dataloader import DataLoader
from utils.misc import set_global_seeds, make_arg_list

MARGINS = [0., 0.1, 0.2, 0.3, 0.4, 0.5]


def find_best_margin(args):
    """ return `best_margin / 0.1` """
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)

    results = []
    for margin in MARGINS:
        model = Perceptron(feature_dim=X_train.shape[-1], margin=margin)
        model.fit(X_train, y_noisy)
        results.append(model.score(X_test, y_test))
    if 'verbose' in args.keys():
        for margin, res in zip(MARGINS, results):
            print(f'margin:{margin}\ttest results:{res}')
    return np.argmax(results)


def run_pam(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)
    model = Perceptron(feature_dim=X_train.shape[-1], margin=args['margin'])
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def run(args):
    logger.configure(f'logs/pam/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    logger.info(args)

    pool = mp.Pool(mp.cpu_count())
    pam_arg = args.copy()

    if 'margin' not in pam_arg.keys():
        best_margin = pool.map(find_best_margin, make_arg_list(pam_arg))
        best_margin = MARGINS[np.bincount(best_margin).argmax()]
        logger.record_tabular('[PAM] best margin', best_margin)
        pam_arg['margin'] = best_margin

    results_pam = pool.map(run_pam, make_arg_list(pam_arg))

    logger.record_tabular('[PAM] accuracy mean', np.mean(results_pam))
    logger.record_tabular('[PAM] accuracy max', np.max(results_pam))
    logger.record_tabular('[PAM] accuracy min', np.min(results_pam))
    logger.record_tabular('[PAM] accuracy std', np.std(results_pam))
    logger.dump_tabular()


if __name__ == '__main__':
    from utils.parser import parse_args
    run(parse_args().__dict__)
