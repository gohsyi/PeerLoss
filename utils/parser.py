import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='heart', choices=[
        'heart', 'breast', 'breast2', 'german', 'banana', 'image', 'thyroid',
        'titanic', 'splice', 'twonorm', 'waveform', 'flare-solar', 'diabetes',
    ])
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--e0', type=float, default=0.2,
                        help='error rate for class 0 (default: 0.2)')
    parser.add_argument('--e1', type=float, default=0.2,
                        help='error rate for class 1 (default: 0.2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--margin', type=float, default=0.,
                        help='margin for PAM (default: 0.)')
    parser.add_argument('--C1', type=float, default=1.,
                        help='C1 for C-SVM (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout for neural networks (default: 0)')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh', 'elu', 'relu6'],
                        help='activation function (default: relu)')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'mse', 'logistic', 'l1', 'huber'],
                        help='loss function (default: bce)')
    parser.add_argument('--hidsize', nargs='+', default=[32, 32], type=int,
                        help='sizes of hidden layers (default: 32, 32)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='least training episodes (default: 1000)')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='batchsize for neural networks (default: 4)')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='proportion of validation set (default: 0.1)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='proportion of test set (default: 0.2)')
    parser.add_argument('--alpha', type=float, default=1.,
                        help='coefficient of peer loss (default: 1)')
    parser.add_argument('--equalize-prior', action='store_true', default=False,
                        help='whether to equalize P(y=1) and P(y=0) (default: False)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='whether to normalize the data (default: True)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to output more information (default: False)')
    args = parser.parse_args()
    return args
