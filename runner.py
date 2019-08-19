from utils.parser import parse_args
from utils.dataloader import DataLoader
from utils.misc import set_global_seeds, make_arg_list

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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


def run(args):
    from run_nn import run as run_nn
    from run_svm import run as run_svm
    from run_pam import run as run_pam

    run_nn(args)
    run_svm(args)
    run_pam(args)

    """
    import numpy as np
    from utils import logger
    
    # logistic regression
    results_lr = pool.map(run_lr, make_arg_list(args))
    logger.record_tabular('[LR] accuracy mean', np.mean(results_lr))
    logger.record_tabular('[LR] accuracy max', np.max(results_lr))
    logger.record_tabular('[LR] accuracy min', np.min(results_lr))
    logger.record_tabular('[LR] accuracy std', np.std(results_lr))
    
    # k nearest neighbours
    results_knn = pool.map(run_knn, make_arg_list(args))
    logger.record_tabular('[KNN] accuracy mean', np.mean(results_knn))
    logger.record_tabular('[KNN] accuracy max', np.max(results_knn))
    logger.record_tabular('[KNN] accuracy min', np.min(results_knn))
    logger.record_tabular('[KNN] accuracy std', np.std(results_knn))
    
    # random forest
    results_rf = pool.map(run_rf, make_arg_list(args))
    logger.record_tabular('[RF] accuracy mean', np.mean(results_rf))
    logger.record_tabular('[RF] accuracy max', np.max(results_rf))
    logger.record_tabular('[RF] accuracy min', np.min(results_rf))
    logger.record_tabular('[RF] accuracy std', np.std(results_rf))
    
    logger.dump_table()
    """


if __name__ == '__main__':
    args = parse_args().__dict__
    run(args)
