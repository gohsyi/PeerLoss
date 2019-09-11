import datetime
import numpy as np
import multiprocessing as mp

from models.perceptron import Perceptron

from utils import logger
from utils.dataloader import DataLoader
from utils.misc import set_global_seeds, make_arg_list

MARGINS = [0, 0.125, 0.25, 0.5, 1, 2, 3, 4, 5]


def find_best_margin(args):
    """ return `best_margin / 0.1` """
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)

    results = []
    for margin in MARGINS:
        model = Perceptron(feature_dim=X_train.shape[-1], margin=margin)
        model.fit(X_train, y_train)
        results.append(model.score(X_val, y_val))
    return results


def run_pam(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    model = Perceptron(feature_dim=X_train.shape[-1], margin=args['margin'])
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def run(args):
    logger.configure(f'logs/{args["dataset"]}/pam/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    logger.info(args)

    pool = mp.Pool(mp.cpu_count())
    pam_arg = args.copy()

    if 'margin' not in pam_arg.keys():
        best_margin = pool.map(find_best_margin, make_arg_list(pam_arg, seeds=8))
        best_margin = np.mean(best_margin, 0)
        if 'verbose' in pam_arg.keys() and pam_arg['verbose']:
            for i in range(len(best_margin)):
                logger.record_tabular(f'[PAM] margin = {MARGINS[i]}', best_margin[i])
            logger.dump_tabular()
        best_margin = MARGINS[best_margin.argmax()]
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
