import datetime
import numpy as np
import multiprocessing as mp

from sklearn.svm import SVC

from utils import logger
from utils.dataloader import DataLoader
from utils.misc import set_global_seeds, make_arg_list

CLASS_WEIGHTS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]


def find_best_c1(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)
    results = []
    for c1 in CLASS_WEIGHTS:
        model = SVC(gamma='auto', class_weight={0: 1., 1: c1})
        model.fit(X_train, y_noisy)
        results.append(model.score(X_test, y_test))
    return np.argmax(results)


def run_c_svm(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, y_noisy, y_test = dataset.prepare_train_test(args)
    model = SVC(gamma='auto', class_weight={0: 1., 1: args['C1']})
    model.fit(X_train, y_noisy)
    return model.score(X_test, y_test)


def run(args):
    logger.configure(f'logs/svm/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    logger.info(args)

    pool = mp.Pool(mp.cpu_count())
    svm_arg = args.copy()

    if 'C1' not in svm_arg.keys():
        best_c1 = pool.map(find_best_c1, make_arg_list(svm_arg))
        best_c1 = CLASS_WEIGHTS[np.bincount(best_c1).argmax()]
        logger.record_tabular('[C-SVM] best C1', best_c1)
        svm_arg['C1'] = best_c1

    results_svm = pool.map(run_c_svm, make_arg_list(svm_arg))

    logger.record_tabular('[C-SVM] accuracy mean', np.mean(results_svm))
    logger.record_tabular('[C-SVM] accuracy max', np.max(results_svm))
    logger.record_tabular('[C-SVM] accuracy min', np.min(results_svm))
    logger.record_tabular('[C-SVM] accuracy std', np.std(results_svm))
    logger.dump_tabular()


if __name__ == '__main__':
    from utils.parser import parse_args
    run(parse_args().__dict__)
