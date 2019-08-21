import datetime
import numpy as np
import multiprocessing as mp

from sklearn.svm import SVC

from utils import logger
from utils.dataloader import DataLoader
from utils.misc import set_global_seeds, make_arg_list

CLASS_WEIGHTS = [0.1, 0.2, 0.25, 0.33, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]


def find_best_c1(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    results = []
    for c1 in CLASS_WEIGHTS:
        model = SVC(gamma='auto', class_weight={0: 1., 1: c1})
        model.fit(X_train, y_train)
        results.append(model.score(X_val, y_val))
    return results


def run_c_svm(args):
    set_global_seeds(args['seed'])
    dataset = DataLoader(args['dataset'])
    X_train, X_test, X_val, y_train, y_test, y_val = dataset.prepare_train_test_val(args)
    model = SVC(gamma='auto', class_weight={0: 1., 1: args['C1']})
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def run(args):
    logger.configure(f'logs/svm/{datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")}')
    logger.info(args)

    pool = mp.Pool(mp.cpu_count())
    svm_arg = args.copy()

    if 'C1' not in svm_arg.keys():
        best_c1 = pool.map(find_best_c1, make_arg_list(svm_arg, seeds=8))
        best_c1 = np.mean(best_c1, 0)
        if 'verbose' in svm_arg.keys() and svm_arg['verbose']:
            for i in range(len(best_c1)):
                logger.record_tabular(f'[C-SVM] C1 = {CLASS_WEIGHTS[i]}', best_c1[i])
            logger.dump_tabular()
        best_c1 = CLASS_WEIGHTS[best_c1.argmax()]
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
