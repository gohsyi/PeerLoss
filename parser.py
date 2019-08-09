import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['heart', 'breast', 'german', 'banana'], type=str, default='heart')
    parser.add_argument('--e0', type=float, default=0.2)
    parser.add_argument('--e1', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidsize', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--test-size', type=float, default=0.25)
    parser.add_argument('--peer-loss', action='store_true', default=False)
    args = parser.parse_args()
    return args
