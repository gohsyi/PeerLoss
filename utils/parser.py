import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='heart', choices=[
        'heart', 'breast', 'breast2', 'german', 'banana', 'image', 'thyroid',
        'titanic', 'splice', 'twonorm', 'waveform', 'flare-solar', 'diabetes',
        'susy', 'higgs',
    ])

    # error rate
    parser.add_argument('--e0', type=float, default=0,
                        help='error rate for class 0 (default: 0)')
    parser.add_argument('--e1', type=float, default=0,
                        help='error rate for class 1 (default: 0)')

    # neural network hyper-parameters
    parser.add_argument('--hidsize', nargs='+', type=int,
                        default=[8, 16, 32, 64],
                        help='sizes of hidden layers for grid search')
    parser.add_argument('--lr', nargs='+', required=True, type=float,
                        default=[0.0007, 0.001, 0.005, 0.01, 0.05],
                        help='learning rates for grid search')
    parser.add_argument('--batchsize', nargs='+', required=True, type=int,
                        default=[1, 4, 16, 32, 64],
                        help='batchsize for neural networks')
    parser.add_argument('--batchsize-peer', nargs='+', required=True, type=int,
                        default=[1, 4, 16, 32, 64],
                        help='batchsize for neural networks')

    # hyper-parameters
    parser.add_argument('--alpha', nargs='+', type=float,
                        default=[-5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0],
                        help='coefficient of peer loss for grid search')
    parser.add_argument('--margin', type=float, default=0.,
                        help='margin for PAM (default: 0.)')
    parser.add_argument('--C1', type=float, default=1.,
                        help='C1 for C-SVM (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout for neural networks (default: 0)')

    # functions
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh', 'elu', 'relu6'],
                        help='activation function (default: relu)')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'mse', 'logistic', 'l1', 'huber'],
                        help='loss function (default: bce)')

    # experiment
    parser.add_argument('--seeds', type=int, default=1,
                        help='repeat experiments across how many seeds')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='least training episodes (default: 1000)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='proportion of validation set (default: 0.15)')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='proportion of test set (default: 0.15)')
    parser.add_argument('--equalize-prior', action='store_true', default=False,
                        help='whether to equalize P(y=1) and P(y=0) (default: False)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='whether to normalize the data (default: True)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to output more information (default: False)')

    args = parser.parse_args()

    args.hidsize = [[hdim, hdim] for hdim in args.hidsize]

    return args
